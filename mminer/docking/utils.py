"""
Utility scripts for post-docking analysis and 3D visualization.
"""

from html import escape as _html_escape
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

__all__ = ["calculate_rmsd", "plot_3d"]

_3DMOL_CDN = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"


class _HTMLView:
    """Lightweight Jupyter-displayable HTML container (renders cleanly via _repr_html_)."""

    def __init__(self, html_str: str):
        self.data = html_str

    def _repr_html_(self) -> str:
        return self.data


def calculate_rmsd(
    ref_mol: str | Path,
    query_mol: str | Path,
    query_num: int = 0,
    add_hydrogen: bool = False,
) -> float:
    """
    Calculate heavy-atom (or all-atom) RMSD between two molecular structures in .sdf format.
    Assumes ``ref_mol`` contains the reference molecule as its first entry.

    :param ref_mol: str or Path
        Filepath to the reference ligand in .sdf format.
    :param query_mol: str or Path
        Filepath to the query/docked poses in .sdf format.
    :param query_num: int
        0-based index of the pose in ``query_mol`` to compare. Default is 0.
    :param add_hydrogen: bool
        If True, add explicit 3D hydrogens to both molecules before calculating RMSD.
    :return: float
        Best symmetry-aware RMSD value in Angstroms.
    """
    ref = Chem.SDMolSupplier(str(ref_mol))
    query = Chem.SDMolSupplier(str(query_mol))

    if len(ref) == 0 or ref[0] is None:
        raise ValueError(f"Failed to read reference molecule from: {ref_mol}")
    if query_num < -len(query) or query_num >= len(query):
        raise IndexError(
            f"query_num {query_num} is out of range for {len(query)} molecule(s) in {query_mol}"
        )

    mol1 = ref[0]
    mol2 = query[query_num]
    if mol2 is None:
        raise ValueError(f"Failed to read query molecule at index {query_num} from: {query_mol}")

    if add_hydrogen:
        mol1 = Chem.AddHs(mol1, addCoords=True)
        mol2 = Chem.AddHs(mol2, addCoords=True)

    return float(AllChem.GetBestRMS(mol1, mol2))


def _load_ligand_pose(
    ligands: str | Path | Chem.Mol,
    pose: int | str = 0,
    remove_hs: bool = False,
) -> Chem.Mol:
    """Load a single 3D RDKit Mol from an SDF path or return an existing RDKit Mol."""
    if isinstance(ligands, Chem.Mol):
        mol = Chem.RemoveHs(ligands) if remove_hs else ligands
        if mol.GetNumConformers() == 0:
            raise ValueError("Input RDKit Mol has no 3D conformer to display.")
        return mol

    lig_path = Path(ligands)
    if not lig_path.exists():
        raise FileNotFoundError(f"Ligand SDF file not found: {lig_path}")

    suppl = Chem.SDMolSupplier(str(lig_path), removeHs=remove_hs)
    n_mols = len(suppl)
    if n_mols == 0:
        raise ValueError(f"No molecules found in ligand file: {lig_path}")

    if isinstance(pose, int):
        if pose < -n_mols or pose >= n_mols:
            raise IndexError(f"Pose index {pose} out of range for {n_mols} molecule(s) in {lig_path}")
        mol = suppl[pose]
        if mol is None:
            raise ValueError(f"Failed to parse molecule at index {pose} in {lig_path}")
    elif isinstance(pose, str):
        mol = None
        for candidate in suppl:
            if candidate is not None and candidate.HasProp("_Name") and candidate.GetProp("_Name") == pose:
                mol = candidate
                break
        if mol is None:
            raise KeyError(f"Pose named {pose!r} not found in {lig_path}")
    else:
        raise TypeError(f"pose must be an int index or str name, got {type(pose).__name__}")

    if mol.GetNumConformers() == 0:
        raise ValueError(f"Selected pose {pose!r} has no 3D conformer to display.")
    return mol


def _js_string(value: str) -> str:
    """Serialize a Python string safely for embedding inside an HTML <script> tag."""
    return json.dumps(value).replace("</", "<\\/")


def plot_3d(
    protein: str | Path,
    ligands: str | Path | Chem.Mol,
    pose: int | str = 0,
    width: int = 650,
    height: int = 600,
    pocket_radius: float = 5.0,
    show_pocket_residues: bool = True,
    show_surface: bool = False,
    remove_hs: bool = False,
) -> _HTMLView:
    """
    Display an interactive 3D protein-ligand docking pose in a Jupyter notebook using 3Dmol.js
    (no ProLIF or extra Python viewer packages required).

    :param protein: str or Path
        Filepath to the protein structure (.pdb, .pdbqt, or .cif/.mmcif).
    :param ligands: str, Path, or rdkit.Chem.Mol
        Filepath to docked poses in .sdf format, or an RDKit Mol with 3D coordinates.
    :param pose: int or str
        0-based pose index (default 0) or exact pose ``_Name`` string in the .sdf file.
    :param width: int
        Viewer width in pixels. Default is 650.
    :param height: int
        Viewer height in pixels. Default is 600.
    :param pocket_radius: float
        Distance cutoff in Angstroms around the ligand used to display nearby binding-site
        residues as sticks (and optional pocket surface). Default is 5.0.
    :param show_pocket_residues: bool
        If True, display protein residues within ``pocket_radius`` Angstroms of the ligand as
        sticks. Default is True.
    :param show_surface: bool
        If True, overlay a translucent molecular surface on the binding pocket. Default is False.
    :param remove_hs: bool
        If True, hide hydrogens on the ligand when loading from SDF. Default is False.
    :return: _HTMLView
        Interactive 3Dmol.js viewer implementing ``_repr_html_()`` for Jupyter notebooks.
    """
    protein_path = Path(protein)
    if not protein_path.exists():
        raise FileNotFoundError(f"Protein file not found: {protein_path}")

    protein_text = protein_path.read_text(encoding="utf-8")
    ext = protein_path.suffix.lower().lstrip(".")
    if ext == "mmcif":
        protein_fmt = "cif"
    elif ext in {"pdb", "pdbqt", "cif", "mol2"}:
        protein_fmt = ext
    else:
        protein_fmt = "pdb"

    lig_mol = _load_ligand_pose(ligands, pose=pose, remove_hs=remove_hs)
    lig_sdf = Chem.MolToMolBlock(lig_mol)

    pose_name = lig_mol.GetProp("_Name").strip() if lig_mol.HasProp("_Name") else ""
    vina_score = lig_mol.GetProp("VINA RESULT").strip() if lig_mol.HasProp("VINA RESULT") else ""

    badge_parts = []
    if pose_name:
        badge_parts.append(_html_escape(pose_name))
    if vina_score:
        badge_parts.append(f"Vina: {_html_escape(vina_score)} kcal/mol")
    badge_html = (
        f'<div class="badge">{" &nbsp;|&nbsp; ".join(badge_parts)}</div>' if badge_parts else ""
    )

    inner_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #fff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  #viewer {{ position: relative; width: 100%; height: 100%; }}
  .badge {{ position: absolute; top: 8px; left: 8px; z-index: 10; background: rgba(255,255,255,0.88); border: 1px solid #d0d7de; border-radius: 4px; padding: 4px 8px; font-size: 12px; color: #24292f; pointer-events: none; }}
</style>
<script src="{_3DMOL_CDN}"></script>
</head>
<body>
{badge_html}
<div id="viewer"></div>
<script>
(function() {{
  const proteinData = {_js_string(protein_text)};
  const proteinFmt = {_js_string(protein_fmt)};
  const ligandData = {_js_string(lig_sdf)};
  const ligandLabel = {_js_string(pose_name or "LIG")};
  const pocketRadius = {float(pocket_radius)};
  const showPocket = {json.dumps(bool(show_pocket_residues))};
  const showSurface = {json.dumps(bool(show_surface))};

  const viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "white" }});
  viewer.addModel(proteinData, proteinFmt);
  viewer.addModel(ligandData, "sdf");

  viewer.setStyle({{ model: 0 }}, {{ cartoon: {{ color: "#c9d1d9", style: "edged" }} }});
  if (showPocket) {{
    viewer.addStyle(
      {{ model: 0, within: {{ distance: pocketRadius, sel: {{ model: 1 }} }}, byres: true }},
      {{ stick: {{ radius: 0.14 }} }}
    );
  }}
  if (showSurface) {{
    viewer.addSurface(
      $3Dmol.SurfaceType.VDW,
      {{ opacity: 0.5, color: "white" }},
      {{ model: 0, within: {{ distance: pocketRadius, sel: {{ model: 1 }} }}, byres: true }}
    );
  }}

  viewer.setStyle({{ model: 1 }}, {{ stick: {{ colorscheme: "cyanCarbon", radius: 0.22 }} }});

  viewer.setHoverable({{}}, true,
    function(atom, v) {{
      if (!atom.label) {{
        const txt = atom.model === 1
          ? ligandLabel + ":" + atom.atom + atom.serial
          : (atom.resn || "") + (atom.resi || "") + (atom.chain ? "." + atom.chain : "") + ":" + atom.atom;
        atom.label = v.addLabel(txt, {{
          position: atom,
          backgroundColor: "rgba(245, 255, 250, 0.92)",
          fontColor: "black",
          fontSize: 12
        }});
      }}
    }},
    function(atom, v) {{
      if (atom.label) {{
        v.removeLabel(atom.label);
        delete atom.label;
      }}
    }}
  );

  viewer.zoomTo({{ model: 1 }});
  viewer.render();
}})();
</script>
</body>
</html>"""

    iframe = (
        f'<iframe srcdoc="{_html_escape(inner_html, quote=True)}" '
        f'style="width:{int(width)}px;height:{int(height)}px;border:1px solid #d0d7de;border-radius:6px;" '
        f'loading="lazy"></iframe>'
    )
    return _HTMLView(iframe)
