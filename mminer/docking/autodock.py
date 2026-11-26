"""
Scripts for Docking
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import warnings
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from mminer._optional import require

__all__ = ["Vina"]

_VINA_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _meeko():
    """Load meeko's PDBQT helpers. meeko lives behind the 'docking' extra."""
    meeko = require("meeko", "docking", "Ligand preparation")
    return (
        meeko.MoleculePreparation,
        meeko.PDBQTWriterLegacy,
        meeko.PDBQTMolecule,
        meeko.RDKitMolCreate,
    )


class Vina:
    def __init__(self, sf_name: str = "vina", verbosity: int = 0, cpu: int = 0, seed: int = 42):
        """
        Initialize the Vina class.
        :param sf_name: str
            Set the scoring function to use. Can be 'vina' or 'ad4'. Defaults to 'vina'.
        :param verbosity: int
            Set how much information to give during docking. 0 - no output. 1 - normal. 2 - verbose.
        :param cpu: int
            Set the number of CPU to use. Defaults to 0 (use all).
        :param seed: int
            Set the Random seed.
        """
        self.sf_name = sf_name
        self.verbosity = verbosity
        self.cpu = cpu
        self.seed = seed

        # Hold prepared compounds and docking results across workflow steps.
        self.compound_id: list[Any] | None = None
        self.compound_smiles: list[str] | None = None
        self.compounds_for_docking: list[str] | None = None
        self.docking_poses: list[_DockedPoseRecord] | None = None
        self.centroid_coords: Any = None

    def prep_receptor(
        self,
        receptor: str | Path | None = None,
        savepath: str | Path | None = None,
        allow_bad_res: bool = True,
    ) -> str:
        """
        Prepare the receptor for docking using Meeko (``mk_prepare_receptor``) and save it as a
        persistent ``.pdbqt`` file at ``savepath``.

        :param receptor: str or Path
            Filepath to the input receptor structure (``.pdb`` or ``.pqr``).
        :param savepath: str or Path
            Output filepath for the prepared receptor (``.pdbqt``).
        :param allow_bad_res: bool
            If True (default), pass ``-a`` to Meeko so waters or unrecognized non-standard
            residues are skipped rather than aborting preparation.
        :return: str
            Path to the saved receptor ``.pdbqt`` file.
        """
        if receptor is None and savepath is None:
            raise ValueError("Need to give filepath for receptor and savepath!")
        elif receptor is None:
            raise ValueError("Need to designate receptor path!")
        elif savepath is None:
            raise ValueError("Need to designate savepath path!")

        receptor_path = Path(receptor)
        if not receptor_path.exists():
            raise FileNotFoundError(f"Receptor file not found: {receptor_path}")

        require("meeko", "docking", "Receptor preparation")

        out_path = Path(savepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        read_flag = "--read_pqr" if receptor_path.suffix.lower() == ".pqr" else "--read_pdb"
        command = [
            sys.executable,
            "-m",
            "meeko.cli.mk_prepare_receptor",
            read_flag,
            str(receptor_path),
            "-p",
            str(out_path),
        ]
        if allow_bad_res:
            command.append("-a")

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not out_path.exists():
            details = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"Meeko receptor preparation failed for '{receptor_path}' "
                f"(exit code {result.returncode}):\n{details}"
            )

        if result.stdout.strip():
            print(result.stdout.strip())
        return str(out_path)

    def prep_compound(
        self,
        input_df: str | Path | pd.DataFrame | None = None,
        smi_col: str | None = None,
        name_col: str | None = None,
        includedFingerprints: bool = False,
    ) -> list[str]:
        """
        Prepare compounds in memory for docking (zero .pdbqt files written to disk).
        Multi-fragment salt SMILES are stripped to the largest heavy-atom fragment for both
        docking and the stored SMILES metadata.

        :param input_df: str, Path, or pd.DataFrame
            Path to a .csv file or a pandas DataFrame containing SMILES and compound name columns.
        :param smi_col: str
            Name of the column with SMILES strings.
        :param name_col: str
            Name of the column with compound names/identifiers.
        :param includedFingerprints: bool
            Deprecated/retained for backwards compatibility.
        :return: list[str]
            Prepared ligand PDBQT strings held in memory.
        """
        del includedFingerprints  # Retained in signature for backwards compatibility.

        if isinstance(input_df, (str, Path)):
            df = pd.read_csv(input_df)
        elif isinstance(input_df, pd.DataFrame):
            df = input_df
        else:
            print(type(input_df))
            raise ValueError(
                "Input should be in .csv format containing at least two columns - smiles and compound name!"
            )

        if smi_col is None and name_col is None:
            raise ValueError("Need a smiles_col and name_col input!")
        elif smi_col is None:
            raise ValueError("smiles_col cannot be None!")
        elif name_col is None:
            raise ValueError("name_col cannot be None!")

        if smi_col not in df.columns:
            raise ValueError(f"The column '{smi_col}' is not found in the DataFrame.")
        if name_col not in df.columns:
            raise ValueError(f"The column '{name_col}' is not found in the DataFrame.")

        self.compounds_for_docking = []
        self.compound_id = []
        self.compound_smiles = []

        MoleculePreparation, PDBQTWriterLegacy, _, _ = _meeko()

        name_list = df[name_col].tolist()
        smi_list = df[smi_col].tolist()

        for smi, name in tqdm(
            zip(smi_list, name_list), total=len(smi_list), desc="Preparing Compounds"
        ):
            if not isinstance(smi, str) or not smi.strip():
                warnings.warn(
                    f"Missing or empty SMILES for compound '{name}'. Skipping.",
                    stacklevel=2,
                )
                continue

            raw_smi = smi.strip()
            mol = Chem.MolFromSmiles(raw_smi)
            if mol is None:
                warnings.warn(
                    f"Invalid SMILES for compound '{name}': {raw_smi!r}. Skipping.",
                    stacklevel=2,
                )
                continue

            # Strip salts by keeping only the largest fragment by heavy-atom count.
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if len(frags) > 1:
                mol = max(frags, key=lambda frag: frag.GetNumHeavyAtoms())
                clean_smi = _mol_to_clean_smiles(mol)
            elif "[H]" in raw_smi:
                clean_smi = _mol_to_clean_smiles(mol)
            else:
                clean_smi = raw_smi

            mol_h = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = self.seed
            result = AllChem.EmbedMolecule(mol_h, params)
            if result != 0:
                params_rand = AllChem.ETKDGv3()
                params_rand.randomSeed = self.seed
                params_rand.useRandomCoords = True
                result = AllChem.EmbedMolecule(mol_h, params_rand)
            if result != 0:
                warnings.warn(
                    f"3D embedding failed for compound '{name}' (SMILES: {clean_smi}). Skipping.",
                    stacklevel=2,
                )
                continue

            preparator = MoleculePreparation()
            mol_setup = preparator.prepare(mol_h)
            for setup in mol_setup:
                pdbqt_str, is_ok, err_msg = PDBQTWriterLegacy.write_string(setup)
                if not is_ok:
                    warnings.warn(
                        f"Meeko PDBQT conversion failed for compound '{name}': {err_msg}. Skipping.",
                        stacklevel=2,
                    )
                    continue
                self.compounds_for_docking.append(pdbqt_str)
                self.compound_id.append(name)
                self.compound_smiles.append(clean_smi)

        return self.compounds_for_docking

    def prep_crystal_lig(
        self,
        sdf_file: str | Path | None = None,
        template_smi: str | None = None,
        savepath: str | Path | None = None,
        save_sdf: bool = False,
    ) -> str:
        """
        Prepare the co-crystal ligand and save it as a .pdbqt file on disk.
        Primarily used for redocking/minimization experiments. When ``template_smi`` is provided,
        bond orders from the template SMILES are assigned onto the 3D crystal structure.

        :param sdf_file: str or Path
            The file path for the co-crystal ligand in .sdf format.
        :param template_smi: str, optional
            Canonical SMILES string for the co-crystal ligand to restore missing PDB bond orders.
            If omitted, the input .sdf is sanitized directly.
        :param savepath: str or Path
            Filepath to save the prepared crystal ligand in .pdbqt format.
        :param save_sdf: bool
            Optional. Save the prepared crystal ligand alongside ``savepath`` with a ``.sdf`` suffix.
        :return: str
            Path to the saved .pdbqt file.
        """
        if sdf_file is None and savepath is None:
            raise ValueError("Need to give filepath for sdf_file and savepath!")
        elif sdf_file is None:
            raise ValueError("Need to designate sdf_file path!")
        elif savepath is None:
            raise ValueError("Need to designate savepath path!")

        suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
        if len(suppl) == 0 or suppl[0] is None:
            raise ValueError(f"Failed to read crystal ligand from: {sdf_file}")

        if len(suppl) > 1:
            print("More than 1 molecule found. Only preparing first compound!")

        cry_mol = suppl[0]

        if template_smi is not None:
            try:
                cry_no_h = Chem.RemoveHs(cry_mol, sanitize=False)
            except Exception:
                cry_no_h = cry_mol

            template = Chem.MolFromSmiles(template_smi)
            if template is None:
                raise ValueError(f"Invalid template_smi: {template_smi!r}")

            fixed_mol = AllChem.AssignBondOrdersFromTemplate(template, cry_no_h)
        else:
            fixed_mol = Chem.Mol(cry_mol)

        Chem.SanitizeMol(fixed_mol)
        mol = Chem.AddHs(fixed_mol, addCoords=True)

        MoleculePreparation, PDBQTWriterLegacy, _, _ = _meeko()
        preparator = MoleculePreparation()
        mol_prep = preparator.prepare(mol)

        pdbqt_chunks: list[str] = []
        for setup in mol_prep:
            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                pdbqt_chunks.append(pdbqt_string)
            else:
                warnings.warn(f"Error converting crystal ligand: {error_msg}", stacklevel=2)

        if not pdbqt_chunks:
            raise RuntimeError(f"Failed to convert crystal ligand '{sdf_file}' to PDBQT.")

        pdbqt_text = "\n".join(pdbqt_chunks)
        out_path = Path(savepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            pdbqt_text + ("\n" if not pdbqt_text.endswith("\n") else ""),
            encoding="utf-8",
        )

        if save_sdf:
            Chem.MolToMolFile(mol, str(out_path.with_suffix(".sdf")))

        print("Co-Crystal Ligand is prepared!")
        return str(out_path)

    def get_grid_centroid(self, sdf_file: str | Path | None = None):
        """
        Obtain the XYZ coordinates to set the docking centroid. Returns an RDKit Point3D object
        whose coordinates are accessible as ``centroid.x``, ``centroid.y``, ``centroid.z``.

        :param sdf_file: str or Path
            The filepath to the co-crystal ligand in .sdf format.
        :return: rdkit.Geometry.Point3D
        """
        if sdf_file is None:
            raise ValueError("Need to designate sdf_file path!")

        crystal = next(Chem.SDMolSupplier(str(sdf_file), sanitize=False))
        centroid = Chem.rdMolTransforms.ComputeCentroid(crystal.GetConformer())
        self.centroid_coords = centroid
        return centroid

    def dock(
        self,
        receptor: str | Path | None = None,
        crystal_lig: str | Path | None = None,
        to_dock: list[str] | None = None,
        centroid: Any = None,
        grid_size: int = 20,
        pose_number: int = 3,
        exhaustiveness: int = 8,
        scoring: str | None = None,
        verbosity: int | None = None,
        cpu: int | None = None,
        seed: int | None = None,
        savepath: str | Path | None = None,
    ):
        """
        Dock query ligands into a protein receptor without leaving per-molecule .pdbqt files on disk.

        When ``sdf_savepath`` is given, poses are streamed straight into a single combined .sdf file
        one ligand at a time (constant O(1) pose memory for large libraries) and ``str(sdf_savepath)``
        is returned. When ``sdf_savepath`` is omitted, poses are returned in memory (and stored on
        ``self.docking_poses``) for saving via ``save_dock_results()``.

        :param receptor: str or Path
            File path to the prepared receptor in .pdbqt format.
        :param crystal_lig: str or Path, optional
            File path to the prepared co-crystal ligand (.pdbqt).
        :param to_dock: list[str], optional
            Prepared ligand PDBQT strings. Defaults to ``self.compounds_for_docking``.
        :param centroid: Point3D or (x, y, z) sequence, optional
            Docking box center coordinates. Defaults to ``self.centroid_coords``.
        :param grid_size: int
            Cubic grid box size in Angstroms. Default is 20.
        :param pose_number: int
            Maximum number of poses to retrieve per ligand. Default is 3.
        :param exhaustiveness: int
            Monte Carlo exhaustiveness for docking. Default is 8.
        :param scoring: str, optional
            Scoring function ('vina' or 'ad4'). Defaults to ``self.sf_name``.
        :param verbosity: int, optional
            Vina verbosity (0, 1, or 2). Defaults to ``self.verbosity``.
        :param cpu: int, optional
            Number of CPUs (0 = all). Defaults to ``self.cpu``.
        :param seed: int, optional
            Random seed. Defaults to ``self.seed``.
        :param savepath: str or Path, optional
            Output filepath to stream all docked poses directly into a single combined .sdf file.
        :return: str or list
            Path to the written .sdf file when ``sdf_savepath`` is provided, otherwise a list of
            in-memory pose records for ``save_dock_results()``.
        """
        scoring = self.sf_name if scoring is None else scoring
        verbosity = self.verbosity if verbosity is None else verbosity
        cpu = self.cpu if cpu is None else cpu
        seed = self.seed if seed is None else seed
        if centroid is None:
            centroid = self.centroid_coords
        if to_dock is None:
            to_dock = self.compounds_for_docking

        if receptor is None:
            raise ValueError("Need to specify a receptor .pdbqt file path!")
        if centroid is None:
            raise ValueError("Need to specify centroid coordinates or call get_grid_centroid() first!")
        if to_dock is None:
            raise ValueError("Need to specify compounds in to_dock or call prep_compound() first!")

        if hasattr(centroid, "x") and hasattr(centroid, "y") and hasattr(centroid, "z"):
            center = [float(centroid.x), float(centroid.y), float(centroid.z)]
        else:
            center = [float(coord) for coord in centroid]
            if len(center) != 3:
                raise ValueError(f"centroid must have 3 coordinates (x, y, z), got {len(center)}.")

        if self.compound_id is not None and len(self.compound_id) == len(to_dock):
            names = list(self.compound_id)
        else:
            if self.compound_id is not None:
                warnings.warn(
                    f"Length of to_dock ({len(to_dock)}) does not match prepared compound_id "
                    f"({len(self.compound_id)}); using fallback names ligand_1..ligand_{len(to_dock)}.",
                    stacklevel=2,
                )
            names = [f"ligand_{i}" for i in range(1, len(to_dock) + 1)]

        if self.compound_smiles is not None and len(self.compound_smiles) == len(to_dock):
            smiles_iter: list[str | None] = list(self.compound_smiles)
        else:
            smiles_iter = [None] * len(to_dock)

        vina = require("vina", "vina", "Docking")
        _meeko()

        v = vina.Vina(sf_name=scoring, verbosity=verbosity, cpu=cpu, seed=seed)
        v.set_receptor(str(receptor))
        v.compute_vina_maps(center=center, box_size=[grid_size, grid_size, grid_size])

        if crystal_lig is not None:
            if isinstance(crystal_lig, str) and ("ROOT" in crystal_lig or "\n" in crystal_lig):
                v.set_ligand_from_string(crystal_lig)
            else:
                v.set_ligand_from_file(str(crystal_lig))

            energy = v.score()
            print(f"Score before minimization: {energy[0]:.3f} (kcal/mol)")
            energy_minimized = v.optimize()
            print(f"Score after minimization : {energy_minimized[0]:.3f} (kcal/mol)")
            print("Docking pose will be minimized by co-crystal ligand")

        streaming = savepath is not None
        out_path: Path | None = None
        writer = None
        if streaming:
            out_path = Path(savepath)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = Chem.SDWriter(str(out_path))

        all_records: list[_DockedPoseRecord] = []

        try:
            for name, compound_pdbqt, smi in tqdm(
                zip(names, to_dock, smiles_iter),
                total=len(names),
                desc="Docking Progress",
            ):
                v.set_ligand_from_string(compound_pdbqt)
                v.dock(exhaustiveness=exhaustiveness, n_poses=pose_number)
                poses_pdbqt = v.poses(n_poses=pose_number)

                records = _pdbqt_poses_to_records(poses_pdbqt, compound_name=str(name), smiles=smi)
                if writer is not None:
                    # Stream directly to disk and let each ligand's RDKit Mols be freed immediately.
                    for rec in records:
                        writer.write(rec.mol)
                else:
                    all_records.extend(records)
        finally:
            if writer is not None:
                writer.close()

        if streaming:
            self.docking_poses = None
            return str(out_path)

        self.docking_poses = all_records
        return all_records

    def save_dock_results(
        self,
        dock_result: list[Any] | None = None,
        sdf_savepath: str | Path | None = None,
    ) -> int:
        """
        Save in-memory docking poses into a single combined .sdf file.

        :param dock_result: list, optional
            List of docking pose records (or legacy (mol, name, score) tuples) obtained from
            ``dock()``. Defaults to ``self.docking_poses``.
        :param sdf_savepath: str or Path
            The filepath to save the combined docking results in .sdf format.
        :return: int
            Number of poses written to the SDF file.
        """
        if dock_result is None:
            dock_result = self.docking_poses
            if dock_result is None:
                raise ValueError(
                    "No in-memory docking poses available! If you passed sdf_savepath to dock(), "
                    "poses were already streamed directly to that .sdf file."
                )
        if sdf_savepath is None:
            raise ValueError("Need to specify sdf_savepath!")

        return _write_records_to_sdf(dock_result, sdf_savepath)


"""Internal support helpers"""


@dataclass(frozen=True)
class _DockedPoseRecord:
    """In-memory representation of a single docked ligand pose."""

    mol: Any  # rdkit.Chem.Mol
    name: str  # "{compound_name}_{pose_rank}"
    smiles: str
    affinity: float
    pose_rank: int
    rmsd_lb: float = 0.0
    rmsd_ub: float = 0.0


def _has_pdbqt_structure(lines: list[str]) -> bool:
    """Return True if a PDBQT chunk contains atom or tree records."""
    return any(line.startswith(("ATOM", "HETATM", "ROOT")) for line in lines)


def _split_pdbqt_models(pdbqt_text: str) -> list[tuple[str, float, float, float]]:
    """Split an in-memory multi-model PDBQT string into (model_pdbqt, affinity, rmsd_lb, rmsd_ub)."""
    models: list[tuple[str, float, float, float]] = []
    cur_lines: list[str] = []
    in_model = False
    affinity = rmsd_lb = rmsd_ub = 0.0

    for line in pdbqt_text.splitlines(keepends=True):
        if line.startswith("MODEL"):
            if in_model and _has_pdbqt_structure(cur_lines):
                models.append(("".join(cur_lines), affinity, rmsd_lb, rmsd_ub))
            cur_lines = []
            in_model = True
            affinity = rmsd_lb = rmsd_ub = 0.0
        elif line.startswith("REMARK VINA RESULT:"):
            nums = _VINA_NUM_RE.findall(line)
            if len(nums) >= 1:
                affinity = float(nums[0])
            if len(nums) >= 2:
                rmsd_lb = float(nums[1])
            if len(nums) >= 3:
                rmsd_ub = float(nums[2])
            cur_lines.append(line)
        elif line.startswith("ENDMDL"):
            if _has_pdbqt_structure(cur_lines):
                models.append(("".join(cur_lines), affinity, rmsd_lb, rmsd_ub))
            cur_lines = []
            in_model = False
            affinity = rmsd_lb = rmsd_ub = 0.0
        else:
            cur_lines.append(line)

    if _has_pdbqt_structure(cur_lines) and (not models or in_model):
        models.append(("".join(cur_lines), affinity, rmsd_lb, rmsd_ub))
    return models


def _mol_to_clean_smiles(mol: Chem.Mol) -> str:
    """Derive a clean hydrogen-suppressed canonical SMILES from an RDKit Mol."""
    try:
        return str(Chem.MolToSmiles(Chem.RemoveHs(mol)))
    except Exception:
        return str(Chem.MolToSmiles(mol))


def _set_pose_props(
    mol: Chem.Mol,
    name: str,
    smiles: str,
    affinity: float,
    pose_rank: int,
    rmsd_lb: float,
    rmsd_ub: float,
) -> None:
    """Apply standardized MolMiner docking SDF properties and strip Meeko's internal tag."""
    if mol.HasProp("meeko"):
        mol.ClearProp("meeko")
    mol.SetProp("_Name", name)
    mol.SetProp("SMILES", smiles)
    mol.SetProp("VINA RESULT", f"{affinity:.3f}")
    mol.SetProp("POSE_RANK", str(pose_rank))
    mol.SetProp("RMSD_LB", f"{rmsd_lb:.3f}")
    mol.SetProp("RMSD_UB", f"{rmsd_ub:.3f}")


def _pdbqt_poses_to_records(
    pdbqt_text: str,
    compound_name: str,
    smiles: str | None = None,
) -> list[_DockedPoseRecord]:
    """Convert an in-memory docked PDBQT string to _DockedPoseRecord objects via Meeko."""
    _, _, PDBQTMolecule, RDKitMolCreate = _meeko()

    resolved_smiles = smiles.strip() if isinstance(smiles, str) and smiles.strip() else None
    records: list[_DockedPoseRecord] = []

    for rank, (model_str, affinity, rmsd_lb, rmsd_ub) in enumerate(
        _split_pdbqt_models(pdbqt_text), start=1
    ):
        try:
            pmol = PDBQTMolecule(model_str, skip_typing=True)
            rdmols = RDKitMolCreate.from_pdbqt_mol(pmol)
        except Exception as exc:
            warnings.warn(
                f"Meeko conversion failed for {compound_name} pose {rank}: {exc}",
                stacklevel=2,
            )
            continue

        if not rdmols or rdmols[0] is None:
            warnings.warn(
                f"Meeko produced no RDKit molecule for {compound_name} pose {rank}.",
                stacklevel=2,
            )
            continue

        mol = rdmols[0]
        pose_smiles = resolved_smiles if resolved_smiles is not None else _mol_to_clean_smiles(mol)
        pose_name = f"{compound_name}_{rank}"
        _set_pose_props(mol, pose_name, pose_smiles, affinity, rank, rmsd_lb, rmsd_ub)

        records.append(
            _DockedPoseRecord(
                mol=mol,
                name=pose_name,
                smiles=pose_smiles,
                affinity=affinity,
                pose_rank=rank,
                rmsd_lb=rmsd_lb,
                rmsd_ub=rmsd_ub,
            )
        )
    return records


def _write_records_to_sdf(records: Iterable[Any], sdf_savepath: str | Path) -> int:
    """Write _DockedPoseRecord objects (or legacy (mol, name, score) tuples) into a single .sdf."""
    out_path = Path(sdf_savepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with Chem.SDWriter(str(out_path)) as writer:
        for rec in records:
            if isinstance(rec, _DockedPoseRecord):
                mol = rec.mol
                if not isinstance(mol, Chem.Mol):
                    warnings.warn(
                        f"Skipping non RDKit::ROMol object: {type(mol)}",
                        stacklevel=2,
                    )
                    continue
                _set_pose_props(
                    mol,
                    rec.name,
                    rec.smiles,
                    rec.affinity,
                    rec.pose_rank,
                    rec.rmsd_lb,
                    rec.rmsd_ub,
                )
            else:
                # Backwards compatibility for legacy (mol, name, score) tuples/lists
                mol, name, score = rec
                if not isinstance(mol, Chem.Mol):
                    warnings.warn(
                        f"Skipping non RDKit::ROMol object: {type(mol)}",
                        stacklevel=2,
                    )
                    continue
                if mol.HasProp("meeko"):
                    mol.ClearProp("meeko")
                mol.SetProp("_Name", str(name))
                if not mol.HasProp("SMILES") or not mol.GetProp("SMILES").strip():
                    mol.SetProp("SMILES", _mol_to_clean_smiles(mol))
                mol.SetProp(
                    "VINA RESULT",
                    f"{score:.3f}" if isinstance(score, (int, float)) else str(score),
                )
            writer.write(mol)
            count += 1
    return count
