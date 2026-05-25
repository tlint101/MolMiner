"""
Scripts for post-docking analysis
"""

import pandas as pd
import prolif as plf
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Optional, Literal, Tuple

__all__ = ['calculate_rmsd', 'Interactions']


def calculate_rmsd(ref_mol, query_mol, query_num=0, add_hydrogen=False):
    """
    Calculate RMSD between two molecular structures. Input compounds must be in .sdf format. Assumes ref_mol only has 1
    molecule.

    Parameters:
        ref_mol: The first molecule.
        query_mol: Query molecule. This is the docking pose of the query molecule.
        query_num: The index of molecule to compare. Default is 0. If more molecules associated with the query sdf file,
        adjust the number accordingly.
        add_hydrogen: If true, add hydrogen to molecules before calculating the RMSD.

    Returns:
        float: RMSD value.
    """

    ref = Chem.SDMolSupplier(ref_mol)
    query = Chem.SDMolSupplier(query_mol)

    mol1 = ref[0]  # Reference only has 1 molecule
    mol2 = query[query_num]  # Molecule can change based on input/number of molecules in sdf

    if add_hydrogen:
        # Add hydrogens
        mol1 = Chem.AddHs(mol1)
        mol2 = Chem.AddHs(mol2)

    # Calculate RMSD without alignment
    rmsd = AllChem.GetBestRMS(mol1, mol2)

    return rmsd


class Interactions:
    def __init__(self, protein: Optional[str], ligands: Optional[str]):
        self.protein = protein
        self.ligands = ligands
        self.ifp = None

    @staticmethod
    def show_interactions():
        # return a list of interactions available
        int_list = plf.Fingerprint.list_available(show_hidden=True)
        return int_list

    def to_table(self, protein: Optional[str] = None, ligands: Optional[str] = None,
              interactions: list = None, docking: bool = False, docking_id: str = None):
        """
        Generate protein-ligand interactions table
        :param protein: Optional[str]
            File path to protein in .pdb format.
        :param ligands: Optional[str]
            File path to molecules in .sdf format.
        :param interactions: list[str]
            A list of interactions to be calculated. Must be given as a list. Additional interactions can be displayed
            using the show_interactions() method. Defaults to ["Hydrophobic", "HBDonor", "HBAcceptor"].
        docking: bool
            If True, will extract docking score and append to table.
        docking_id: str
            The ID corresponding to the docking score header. This will depend on the docking program used. Param only
            works if docking param is True.
        :return:
        """
        # set instance variables
        protein = protein or self.protein
        ligands = ligands or self.ligands

        # load pdb and sdf
        p_path = Chem.MolFromPDBFile(protein, removeHs=False)
        prot = plf.Molecule(p_path)
        lig = plf.sdf_supplier(ligands)

        # set default interactions
        interactions = interactions or ["Hydrophobic", "HBDonor", "HBAcceptor"]

        # generate ifp
        fp = plf.Fingerprint(interactions)
        fp.run_from_iterable(lig, prot)
        self.fp = fp

        # convert to table
        df = fp.to_dataframe(count=True)

        # clean up table
        # Swap multi-index level to have interaction type first
        df.columns = df.columns.swaplevel(1, 2)
        df.sort_index(axis=1, level=0, inplace=True)

        # Flatten and join multi-index header
        df.columns = (
            df.columns.map('_'.join)
            .str.strip('_')
            .str.replace("UNL1_", "", regex=False)
            .str.rstrip('.A')  # Remove chain suffix if present
        )

        # Convert to binary
        df = df.astype(int)

        # load .sdf molecules into a PandasDataframe and extract name
        sdf_df = rdkit.Chem.PandasTools.LoadSDF(ligands)
        lig_name = sdf_df['ID']  # table header may change depending on docking program
        lig_name = pd.DataFrame(data=lig_name)
        if docking:
            if docking_id is None:
                lig_name['score'] = sdf_df['VINA RESULT']
            else:
                lig_name['score'] = sdf_df[docking_id]
        lig_name.reset_index(drop=True, inplace=True)

        # OPTIONAL: By default, program will use the same header as the sdf_df.
        lig_name = lig_name.rename(columns={'ID': 'name'}).reset_index(drop=True)

        # combine tables
        frames = [lig_name, df]
        final_df = pd.concat(frames, axis=1, join='inner')
        final_df = final_df.set_index('name')
        final_df.reset_index(drop=False, inplace=True)

        return final_df

    def plot_int(self, pose: int = 0, molsize: int = 35, width: str = "100%", height: str = "500px", rotation: int = 0,
                 carbon: float = 0.16, kekulize=True):
        """
        Plot a protein-ligand interaction in 2D format. Function tested with Jupyter Notebook and includes
        interactivity.
        :param pose: int
            Molecule number from sdf file.
        :param molsize: int
            Modify the size of the displayed molecule.
        :param width: str
            Set the width of the IFrame window.
        :param height: str
            Set the height of the IFrame window.
        :param rotation: int
            Rotate the structure on the XY plane.
        :param carbon: float
            Size the carbon atom dots. Use '0' to hide the carbon dots.
        :param kekulize: bool
            Kekulize the ligand.
        :return:
        """

        pose_index = pose
        lig = plf.sdf_supplier(self.ligands)

        return self.fp.plot_lignetwork(lig[pose_index], kind="frame", molsize=molsize, width=width, height=height,
                                       rotation=rotation, carbon=carbon, frame=pose_index, kekulize=kekulize)


    def plot_barcode(self, figsize: Tuple[int, int] = (8, 10), dpi: int = 300, n_frame_ticks: int = 10,
                     residues_tick_location: Literal["top", "bottom"] = "top", xlabel: str = "Pose",
                     subplots_kwargs: Optional[dict] = None, tight_layout_kwargs: Optional[dict] = None, ):
        """
        Generate a barcode image of the molecule interaction profile.
        :param figsize: Tuple[int, int]
            Size of the matplotlib figure.
        :param dpi: int
            Set the figure DPI.
        :param n_frame_ticks: int
            Set the number of ticks on the X axis.
        :param residues_tick_location: Literal["top", "bottom"]
            Seet the Y ticks to appear at the top or at the bottom of the series of interactions for each residue.
        :param xlabel: str = "Pose"
            Label for the X axis.
        :param subplots_kwargs: Optional[dict]
            Params to be passed to matplotlib.pyplot.subplots.
        :param tight_layout_kwargs:
            Params to be passed to the matplotlib.figure.Figure.tight_layout.
        :return:
        """
        return self.fp.plot_barcode(figsize=figsize, dpi=dpi, n_frame_ticks=n_frame_ticks,
                                    residues_tick_location=residues_tick_location, xlabel=xlabel,
                                    subplots_kwargs=subplots_kwargs, tight_layout_kwargs=tight_layout_kwargs)


    def plot_3d(self, pose=None):
        """
        Generate the protein-ligand interaction in 3D format.
        :param pose: int
            Molecule number from sdf file.
        :return:
        """
        protein = self.protein
        p_path = Chem.MolFromPDBFile(protein, removeHs=False)
        prot = plf.Molecule(p_path)
        lig = plf.sdf_supplier(self.ligands)

        return self.fp.plot_3d(lig[pose], prot, frame=pose, display_all=False)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
