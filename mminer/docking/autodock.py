"""
Scripts for Docking
"""

import re
import os
import sys
import site
import warnings
import subprocess
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools
import vina
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate

__all__ = ['Vina']


class Vina:
    def __init__(self, sf_name='vina', verbosity=0, cpu=0, seed=42):
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

        # To hold compound_id which will be processed in functions below.
        self.compound_id = None
        self.compounds_for_docking = None
        self.docking_poses = None
        self.centroid_coords = None

    def prep_receptor(self, receptor=None, savepath=None):
        """
        Prepare the receptor for docking using AutoDockTools_py3.
        :param receptor: str
            Set the file path for the receptor.
        :param savepath: str
            Set the save path for the prepared receptor.
        :return:
        """
        # get python interpreter path
        python_executable = sys.executable
        site_packages = site.getsitepackages()[0]
        prepare_receptor_script = os.path.join(site_packages, "AutoDockTools/Utilities24/prepare_receptor4.py")

        # set python command
        command = [
            python_executable,
            prepare_receptor_script,
            "-r", receptor,
            "-o", savepath
        ]
        # run python command
        result = subprocess.run(command, capture_output=True, text=True)

        return print(result.stdout)

    def prep_compound(self, input_df: str or pd.DataFrame = None, smi_col: str = None, name_col: str = None,
                      includedFingerprints: bool = False):
        """
        Prepare compounds for docking. Input should be in .csv format containing at least two columns - smiles and name.
        :param input_df: str or pd.DataFrame
            A pandas dataframe table containing at least two columns with smiles string and compound name.
        :param smi_col: str
            Name of the column with smiles strings.
        :param includedFingerprints: bool
            Optional. To include fingerprint with ROMol.
        :return:
        """
        # confirm if input_df is a filepath or a pd.DataFrame
        if isinstance(input_df, str):
            df = pd.read_csv(input_df)
        elif isinstance(input_df, pd.DataFrame):
            df = input_df
        else:
            print(type(input_df))
            raise ValueError(
                'Input should be in .csv format containing at least two columns - smiles and compound name!')

        # in case user does not give smiles_col
        if smi_col is None and name_col is None:
            raise ValueError('Need a smiles_col and name_col input!')
        elif smi_col is None:
            raise ValueError('smiles_col cannot be None!')
        elif name_col is None:
            raise ValueError('name_col cannot be None!')

        # check if input strings are found in DataFrame.
        if smi_col not in df.columns:
            raise ValueError(f"The column '{smi_col}' is not found in the DataFrame.")
        if name_col not in df.columns:
            raise ValueError(f"The column '{name_col}' is not found in the DataFrame.")

        # Convert smi into ROMol
        PandasTools.AddMoleculeColumnToFrame(df, smilesCol=smi_col, molCol='ROMol',
                                             includeFingerprints=includedFingerprints)

        # list to hold docking results and name
        self.compounds_for_docking = []
        self.compound_id = []

        # Convert compound column into list for iteration
        compounds = df['ROMol'].tolist()
        name_list = df[name_col].tolist()
        smi_list = df[smi_col].tolist()

        # there is one molecule in this SD file, this loop iterates just once
        for compound, smi, name in tqdm(zip(compounds, smi_list, name_list), total=len(compounds), desc='Preparing Compounds'):
            mol = Chem.AddHs(compound)
            params = AllChem.ETKDGv3()
            params.randomSeed = self.seed
            result = AllChem.EmbedMolecule(mol, params)
            if result != 0:
                warnings.warn(f"3D embedding failed for SMILES: {smi}. \n SKIPPING!...")
                continue
            preperator = MoleculePreparation()
            mol_setup = preperator.prepare(mol)
            for setup in mol_setup:
                pdbqt_string = PDBQTWriterLegacy.write_string(setup)  # Will return a tuple (3)
                pdbqt_string = pdbqt_string[0]  # To only obtain the string
                self.compounds_for_docking.append(pdbqt_string)
                self.compound_id.append(name)

        return self.compounds_for_docking

    def prep_crystal_lig(self, sdf_file: str = None, template_smi: str = None, savepath: str = None,
                         save_sdf: bool = False):
        """
        Prepare the crystal ligand for docking. Primarily used for redocking experiment. Bond order from PDB file will
        be fixed, but requires smiles string of molecule.
        :param sdf_file: str
            The file path for the co-crystal ligand. It should be in .sdf format.
        :param template_smi: str
            Canonical smiles string for the co-crystal ligand to be used as a template.
        :param savepath: str
            The filepath to save the prepared crystal ligand. It will be in .pdbqt format.
        :param save_sdf: bool
            Optional. To save the prepared crystal ligand in .sdf format with correct bond order.
        :return:
        """
        # Check if params are given
        if sdf_file is None and savepath is None:
            raise ValueError('Need to give filepath for sdf_file and savepath!')
        elif sdf_file is None:
            raise ValueError('Need to designate sdf_file path!')
        elif savepath is None:
            raise ValueError('Need to designate savepath path!')

        # read molecule
        suppl = Chem.SDMolSupplier(sdf_file, sanitize=False)

        if len(suppl) > 1:
            print("More than 1 molecule found. Only preparing first compound!")

        cry_mol = suppl[0]

        # Optional?
        # remove hydrogen if present
        try:
            cry_mol = Chem.RemoveHs(cry_mol)
        except:
            pass

        # read template
        template = Chem.MolFromSmiles(template_smi)

        # assign bond orders to crystal ligand
        fixed_mol = AllChem.AssignBondOrdersFromTemplate(cry_mol, template)

        # Map atoms from the template to the docked pose
        template_to_docked_mapping = cry_mol.GetSubstructMatch(template)

        if not template_to_docked_mapping:
            raise ValueError("Failed to match template to docked pose.")

        # Transfer 3D coordinates from docked_pose to fixed_mol
        new_conformer = Chem.Conformer(fixed_mol.GetNumAtoms())
        docked_conformer = cry_mol.GetConformer()

        # set coordinates of heavy atoms to fixed_mol
        for new_idx, docked_idx in enumerate(template_to_docked_mapping):
            pos = docked_conformer.GetAtomPosition(docked_idx)
            new_conformer.SetAtomPosition(new_idx, pos)

        # Add 3D coordinates to the molecule
        fixed_mol.AddConformer(new_conformer, assignId=True)

        # Add explicit hydrogens, addCoords to add h to coordinates
        mol = Chem.AddHs(fixed_mol, addCoords=True)

        # Prepare the molecule using Meeko
        preparator = MoleculePreparation()
        mol_prep = preparator.prepare(mol)

        # generate pdbqt file
        with open(savepath, 'w') as pdbqt_output:
            for setup in mol_prep:

                pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
                if is_ok:
                    pdbqt_output.write(pdbqt_string + '\n')
                else:
                    print(f"Error converting molecule: {error_msg}")

        # save sdf
        if save_sdf is True:
            # replace pdbqt with sdf extension
            sdf_path = savepath.replace('pdbqt', 'sdf')
            Chem.MolToMolFile(mol, sdf_path)

        # Message to signify ligand is prepped.
        print("Co-Crystal Ligand is prepared!")

        return savepath

    def get_grid_centroid(self, sdf_file: str = None):
        """
        Obtain the XYZ coordinates to set the docking centroid. This will return rdkit object. XYZ coordinates can be
        obtained by setting results to a variable and writing: variable.x, variable.y, variable.z.
        :param sdf_file: str
            The filepath to co-crystal ligand. It should be in .sdf format.
        :return:
        """
        # Check input params
        if sdf_file is None:
            raise ValueError('Need to designate sdf_file path!')

        # read first molecule from .sdf and obtain xyz coord
        crystal = next(Chem.SDMolSupplier(sdf_file, sanitize=False)) # sanitize not needed, only want coordinates
        centroid = Chem.rdMolTransforms.ComputeCentroid(crystal.GetConformer())

        # set instance variable
        self.centroid_coords = centroid

        return centroid

    def dock(self, receptor=None, crystal_lig=None, to_dock=None, centroid=None, grid_size: int = 20,
             pose_number: int = 3, exhaustiveness: int = 8, folder_path: str = None, scoring: str = 'vina',
             verbosity: int = 0, cpu: int = 0, seed: int = 42):
        """
        Dock query ligands into protein receptor.

        :param receptor: str
            File path to the receptor to dock into. It should be in .pdbqt format.
        :param crystal_lig: str (optional)
            File path for the co-crystal ligand. The docking score will take the structure and minimize the score. This
            could help improve docking accuracy, optimization, and validation of binding modes.
        :param to_dock: list
            List of molecules to dock. The list should be in a string format converted from molecules in pdbqt format.
        :param centroid: tuple
            Input the XYZ coordinates of the docking centroid. By default, the centroid will accept an rdkit.Geometry
            object, which contains the 3D coordinates in a tuple. Or userse can input their own coordinates.
        :param grid_size: int
            Set the grid size for docking. Default is 20.
        :param pose_number: int
            Set the number of poses to write for each protein-ligand docking pair.
        :param exhaustiveness: int
            Set the Monte Carlo exhaustiveness for docking. Default is 8.
        :param folder_path: str
            Set the folder path to save the docking results.
        :param scoring: str
            Set the scoring parameters. can use 'Vina' or 'ad4'. Default to vina.
        :param verbosity: int
            Set how much information to display during docking. 0: no output. 1: normal. 2: verbose. Default to 0.
        :param cpu: int
            The number of CPU to use. Default to 0 (all CPUS).
        :param seed: int
            Set the seed for reproducibility. Default to 42.
        :return:
        """

        # ensure default is same as initialized instance
        if verbosity == self.verbosity:
            verbosity = self.verbosity
        if cpu == self.cpu:
            cpu = self.cpu
        if seed == self.seed:
            seed = self.seed
        if centroid is None:
            centroid = self.centroid_coords

        # output mol_list, which contains rdkit mol, name, and score
        v = vina.Vina(sf_name=scoring, verbosity=verbosity, cpu=cpu, seed=seed)
        v.set_receptor(receptor)

        # Set grid and box size
        v.compute_vina_maps(center=[centroid.x, centroid.y,
                                    centroid.z], box_size=[grid_size, grid_size, grid_size])

        # minimize the receptor for ligand docking. Helps with accuracy, optimization, and validation of binding modes.
        if crystal_lig is not None:
            v.set_ligand_from_file(crystal_lig)

            # score co-crystal ligand
            energy = v.score()
            print('Score before minimization: %.3f (kcal/mol)' % energy[0])

            # minimize the score
            energy_minimized = v.optimize()
            print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])

            print('Docking pose will be minimized by co-crystal ligand')

        output_vina = []

        for name, compound in tqdm(zip(self.compound_id, to_dock), total=len(self.compound_id),
                                   desc='Docking Progress'):
            v.set_ligand_from_string(compound)
            v.dock(exhaustiveness=exhaustiveness, n_poses=pose_number)
            if folder_path is None:
                raise ValueError("Need to specify a folder path!")
            else:
                file_path = os.path.join(folder_path, f"{name}.pdbqt")
            # print(file_path) # output for troubleshooting
            v.write_poses(pdbqt_filename=file_path, n_poses=pose_number, overwrite=True)
            output_vina.append(v.poses())

        dock_result_list = _read_and_split_pdbqt(folder_path)

        # Convert docked pdbqt strings into RDMol
        mol_strings = []
        name_list = []
        score_list = []

        for mol_string in dock_result_list:
            if isinstance(mol_string, tuple):
                pdbqt_str = mol_string[0]  # Extract PDBQT string

                # Extract molecule name
                mol_name_pose = mol_string[1]
                name_list.append(mol_name_pose)

                # Extract score
                score = mol_string[2]
                score_list.append(score)

                # Construct the string
                full_string = f'{pdbqt_str}\n{mol_name_pose}\n{score}\n'

                mol_strings.append(full_string)

        # Take string and convert to RDMol
        mol_list = []

        for pdbqt_str, name, score in zip(mol_strings, name_list, score_list):
            pdbqt_mol = PDBQTMolecule(pdbqt_str)
            mol = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
            mol.append(name)
            mol.append(score)
            mol_list.append(mol)

        self.docking_poses = mol_list

        return mol_list

    def save_dock_results(self, dock_result: list = None, sdf_savepath: str = None):
        """
        Save docking poses
        :param dock_result: list
            List of docking poses obtained from the dock method.
        :param sdf_savepath: str
            The filepath to save the docking result in .sdf format.
        :return:
        """

        if dock_result is None:
            dock_result = self.docking_poses
            if dock_result is None:
                raise ValueError('No docking poses given!')

        # Write RDKit molecules to SDF file
        with Chem.SDWriter(sdf_savepath) as writer:
            for mol, name, score in dock_result:
                if isinstance(mol, Chem.Mol):
                    mol.SetProp("_Name", name)  # Set molecule name as a property
                    mol.SetProp("VINA RESULT", score)
                    writer.write(mol)
                else:
                    print(f"Skipping non RDKit::ROMol object: {type(mol)}")


"""Support functions"""


def _read_and_split_pdbqt(folder):
    """
    Support function for docking. Function will read files in a folder containing the molecules in .pdbqt format.
    Molecules will be converted into a list of strings that will be fed directly into AutoDock Vina.
    """
    poses = []
    current_pose_name = None
    for filename in os.listdir(folder):
        if filename.endswith('.pdbqt'):
            compound_name = os.path.splitext(filename)[0]
            with open(os.path.join(folder, filename), 'r') as file:
                lines = file.readlines()
                pose_lines = []
                vina_result = None
                for line in lines:
                    if line.startswith('MODEL'):
                        if pose_lines:
                            poses.append((''.join(pose_lines), current_pose_name, vina_result))
                            pose_lines = []
                            vina_result = None
                        current_pose_name = f'{compound_name}_pose{line.strip().split()[1]}'
                    elif line.startswith('REMARK VINA RESULT:'):
                        vina_result_match = re.search(r'-?\d+\.\d+', line)
                        if vina_result_match:
                            vina_result = vina_result_match.group()
                    elif line.startswith('ENDMDL'):
                        if pose_lines:
                            poses.append((''.join(pose_lines), current_pose_name, vina_result))
                            pose_lines = []
                            vina_result = None
                    else:
                        pose_lines.append(line)
    return poses


def _add_hydrogens(mol):
    """
    Support function to add hydrogens to a molecule
    """
    return Chem.AddHs(mol)
