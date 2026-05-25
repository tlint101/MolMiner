"""
Scripts to draw and highlighter molecular fragments of a molecule.
"""
import itertools
import fsspec
from typing import Optional, List, Union
from collections import defaultdict
import safe as sf
import datamol as dm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rdkit
from rdkit import Chem, Geometry
from rdkit.Chem import rdRGroupDecomposition, rdqueries, rdDepictor, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG, Image
from PIL import Image as pilImage
from io import BytesIO

__all__ = ["Analogs", "RDKitHighlight", "Draw"]


class Draw:
    def __init__(self, mol: str = None, fragments: list[str, dm.Mol or Chem.rdchem.Mol] = None):
        self.mol = mol
        self.fragments = fragments

    def highlight(self,
                  mol: str = None,
                  fragments: Union[str, dm.Mol, Chem.rdchem.Mol, list[Union[str, dm.Mol, Chem.rdchem.Mol]]] = None,
                  mol_size: tuple[int, int] = (300, 300),
                  use_svg: bool = True,
                  style: Optional[str] = 'rdkit',
                  colors: Optional[list[str]] = None,
                  highlight_bond_width_multiplier: int = 12,
                  legend: list[str, None] = None,
                  legend_fontsize: int = 16,
                  savefig: str = None,
                  **kwargs
                  ):
        """
        Highlight matching fragments of a molecule. This method is more flexible and should work for both RDKit or SAFE
        molecules.
        :param mol: str
            Smiles string of query molecule.
        :param fragments: Union[list[str, dm.Mol or Chem.rdchem.Mol]]
            List of fragments to match to query molecule.

        :param mol_size: tuple[int, int]
            Set the molecule size
        :param use_svg: bool
            Set to return image as svg or png
        :param style: str
            Set the image highlighting mode. Can be either 'rdkit', 'lasso', 'fill', or 'color'.
        :param colors: Optional[list[str]
            Give a list of colors for highlighting matching fragments. Can be color name or hex code.
        :param highlight_bond_width_multiplier: int
            The multiplier to use for the bond width. Used in conjunction with the 'fill' mode.
        :param legend: list[str, None]
            Input legends for each molecule. Must be given as a list.
        :param legend_fontsize: int
            Set the legend font size for drawn molecule.
        :param savefig: str
            File path to save the figure
        :param kwargs: Arguments for RDKit drawing function
        :return:
        """
        global color_list_bond
        kwargs["legends"] = legend
        kwargs["legend_fontsize"] = legend_fontsize
        kwargs["mol_size"] = mol_size
        kwargs["use_svg"] = use_svg
        if highlight_bond_width_multiplier is not None:
            kwargs["highlightBondWidthMultiplier"] = highlight_bond_width_multiplier

        if style == "color":
            kwargs["continuousHighlight"] = False
            kwargs["circleAtoms"] = kwargs.get("circleAtoms", False) or False

        # match instance variable
        if mol is None:
            mol = self.mol
        if fragments is None:
            fragments = self.fragments

        # if input is a str or a list of str
        if isinstance(fragments, (str, dm.Mol, Chem.rdchem.Mol)):
            fragments = [fragments]
        elif isinstance(fragments, list):
            fragments = [dm.to_mol(frag, sanitize=False) if isinstance(frag, str) else frag for frag in fragments]
        else:
            raise TypeError("Fragments must be a string, molecule, or list of them.")

        mol = dm.to_mol(mol, remove_hs=False)

        if colors:
            cm = mcolors.ListedColormap(colors)
            frag_number = len(fragments)
            if len(colors) != frag_number:
                raise ValueError(f"Length of color list does not match number of fragments length {len(fragments)}.")
            color_list = [cm(1.0 * i / len(fragments)) for i in range(len(fragments))]
        else:
            cm = plt.get_cmap("gist_rainbow")
            color_list = [cm(1.0 * i / len(fragments)) for i in range(len(fragments))]

        # check to see highlight_types
        highlight_types = ['color', 'fill', 'lasso', 'rdkit']
        if style not in highlight_types:
            raise ValueError(f"Highlight types can only be 'color', 'fill', 'lasso' or 'rdkit'.")

        # if lasso, check for user colors
        elif style == "lasso":
            # the extracted kwargs for legends and legend_fontsize is duplicated for lasso. Will remove them.
            if "legends" in kwargs:
                del kwargs["legends"]
            if "legend_fontsize" in kwargs:
                del kwargs["legend_fontsize"]

            return dm.lasso_highlight_image(target_molecules=mol, search_molecules=fragments, color_list=color_list,
                                            legends=legend, **kwargs)

        # to draw molecules with rdkit shading
        elif style == "rdkit":
            color_list = _convert_colors_to_rgba(color_list)
            color_list_bond = _convert_colors_to_rgba(color_list, bond=True)

        atom_indices = []
        bond_indices = []
        atom_colors = {}
        bond_colors = {}

        for x, frag in enumerate(fragments):
            atom_matches, bond_matches = dm.substructure_matching_bonds(mol, frag)
            atom_matches = list(itertools.chain(*atom_matches))
            bond_matches = list(itertools.chain(*bond_matches))
            atom_indices.extend(atom_matches)
            bond_indices.extend(bond_matches)
            atom_colors.update({atom_idx: color_list[x] for atom_idx in atom_matches})

            # adjust bond color transparency if style set to 'rdkit'
            if style == "rdkit":
                bond_colors.update({bond_idx: color_list_bond[x] for bond_idx in bond_matches})
            else:
                bond_colors.update({bond_idx: color_list[x] for bond_idx in bond_matches})

        # remove select kwargs beefore calling dm.to_image()
        keys_to_remove = [
            "legends",
            "legend_fontsize",
            "mol_size",
            "highlightBondWidthMultiplier",
            "use_svg",
            "continuousHighlight",
            "circleAtoms",
        ]
        # Remove them from kwargs
        for key in keys_to_remove:
            kwargs.pop(key, None)

        # check save figure type:
        if savefig:
            if savefig.endswith(".svg") and use_svg is False:
                raise ValueError("file extension in 'savefig' is '.svg' and is not supported if use_svg set to False!")
            elif savefig.endswith(".png") and use_svg is True:
                raise ValueError("file extension in 'savefig' is '.png' and is not supported if use_svg set to True!")

        return dm.to_image(
            [mol],
            highlight_atom=[atom_indices],
            highlight_bond=[bond_indices],
            highlightAtomColors=[atom_colors],
            highlightBondColors=[bond_colors],
            useBWAtomPalette=True if style == "color" else False,  # for color style
            continuousHighlight=False if style == "color" else True,  # for color style, disable glow effect
            circleAtoms=False if style == "color" else None, # for color style, disable circle atom styles
            outfile=savefig,
            **kwargs,
        )

    def safe_highlight(self,
                       mol: str or sf = None,
                       slicer: str = 'brics',
                       fragments: Union[str, dm.Mol, List[Union[str, dm.Mol]]] = None,
                       mol_size: [[int, int], int] = (300, 300),
                       use_svg: bool = True,
                       style: str = "color",
                       colors: Optional[list[str]] = None,
                       highlight_bond_width_multiplier: int = 12,
                       legend: list[str, None] = None,
                       legend_fontsize: int = 16,
                       savefig: str = None,
                       **kwargs
                       ):
        """
        Highlight fragments using SAFE strings. The molecule will be converted into SAFE string using default BRICS
        slicer. Modify parameters accordingly.
        :param mol: str
            Must be SAFE string
        :param slicer: str
            Set the slicing algorithm. Will default to BRICS
        :param fragments: list[str, dm.Mol or Chem.rdchem.Mol]
            List of fragments to match to query molecule. If None, matching will be performed using fragments from
            default slicing from SAFE.
        :param legend: list[str, None]
            Input legends for each molecule. Must be given as a list.
        :param mol_size: tuple[int, int]
            Set the molecule size
        :param use_svg: bool
            Set to return image as svg or png
        :param style: str
            Set the image highlighting mode. Can be either 'lasso', 'fill', or 'color'.
        :param colors: Optional[list[str]
            Give a list of colors for highlighting matching fragments. Can be color name or hex code.
        :param highlight_bond_width_multiplier: int
            The multiplier to use for the bond width. Used in conjunction with the 'fill' mode.
        :param legend_fontsize: int
            Set the legend font size for drawn molecule.
        :param savefig: str
            File path to save the figure
        :param kwargs: Arguments for RDKit drawing function
        :return:
        """
        kwargs["legends"] = legend
        kwargs["mol_size"] = mol_size
        kwargs["use_svg"] = use_svg
        if highlight_bond_width_multiplier is not None:
            kwargs["highlightBondWidthMultiplier"] = highlight_bond_width_multiplier

        if style == "color":
            kwargs["continuousHighlight"] = False
            kwargs["circleAtoms"] = kwargs.get("circleAtoms", False) or False

        # match instance variable
        if mol is None:
            mol = self.mol
        if fragments is None:
            fragments = self.fragments

        # convert mol smiles into safe string
        safe_str = sf.encode(mol, slicer=slicer)

        if fragments is None:
            fragments = [
                sf.decode(x, as_mol=False, remove_dummies=True, ignore_errors=False)
                for x in safe_str.split(".")
            ]
        elif fragments and len(fragments) > 0:
            parsed_fragments = []
            for fg in fragments:
                if isinstance(fg, str):
                    fg = sf.decode(fg, as_mol=False, remove_dummies=True, ignore_errors=False)
                parsed_fragments.append(fg)
            fragments = parsed_fragments
        else:
            fragments = []

        # convert safe_str into mol
        mol = dm.to_mol(safe_str, remove_hs=False)

        # highlight fragments
        fig = self.highlight(mol=mol, fragments=fragments, mol_size=mol_size, use_svg=use_svg, style=style,
                             colors=colors, highlight_bond_width_multiplier=highlight_bond_width_multiplier,
                             legend=legend, legend_fontsize=legend_fontsize, savefig=savefig)

        return fig

    # support function to draw molecule with atom index
    def atom_number(self, mol: Chem.Mol = None, label: str = "atomNote", size: tuple = (300, 300)):
        """
        Draw query molecule with labeled RDKit atom indices.
        :param mol: Chem.Mol
            A molecule in ROMol format.
        :param label: str
            Determines which style to label the molecule. Defaults to atomNote. In total, can use 'atomNote',
            'atomLabel', and 'molAtomMapNumber'.
        :param size: tuple
            Determine the size of the molecule to draw.
        :return:
        """
        # set instance variable
        if mol is None:
            mol = self.mol

        # Check if 2D coordinates are missing, and compute them if necessary. This will also strip away 3D coordinates.
        if not mol.GetNumConformers() or mol.GetConformer().Is3D():
            AllChem.Compute2DCoords(mol)

        # get atom indices
        for atom in mol.GetAtoms():
            if label == 'atomNote' or label == 'atomLabel' or label == 'molAtomMapNumber':
                atom.SetProp(label, str(atom.GetIdx()))
            else:
                raise ValueError("Only 'atomNote', 'atomLabel', and 'molAtomMapNumber' accepted!")

        # draw molecule
        img = Chem.Draw.MolToImage(mol, size=size)

        return img


# code adapted from Greg Landrum
# https://greglandrum.github.io/rdkit-blog/posts/2021-08-07-rgd-and-highlighting.html

class Analogs:
    def __init__(self, mol: Union[str, list] = None, core: str = None):
        if mol is not None:
            if isinstance(mol, str):
                self.mol = [mol]
            elif isinstance(mol, list):
                self.mol = mol

        if core:
            self.core = core

    def highlight_rgroups(self, mol: Union[str, list] = None, core: str = None, width: int = 350, height: int = 200,
                          colors: Union[str, list] = None, fill_ring: bool = True, legend: Union[str, list] = "",
                          r_labels: tuple = ('R1', 'R2', 'R3', 'R4'), use_svg: bool = True,
                          source_idx_property: str = "SourceAtomIdx", savepath: str = None, return_data: bool = False,
                          query_mol: int = 0):
        """
        Highlight R-Groups of a given molecule/core pair. This is done using RDKit's R-group deomposition tool.
        :param mol: Union[str, list]
            Input molecule. Can be a single smiles string or a list of smiles strings. This function will only draw and
            highlight one molecule. If a list is given, the first molecule in the list will be drawn by default.
            To draw and highlight additional molecules, use the highlight_rgroups_grid() method.
        :param core: str
            Smiles string that will function as the core for the molecules.
        :param width: int
            Set the width of the drawn molecule.
        :param height: int
            Set the height of the drawn molecule.
        :param colors: Union[str, list]
            Set the highlight color palette. If a list is given, can be strings of color names, hex codes, or a
            combination of either. If a string is given, a colorblind palette can be used. Only three types are
            included: 'tol', 'ibm', or 'okabe'. The okabe color palette will be used by default.
        :param fill_ring: bool
            Fill the highlight to give the "bubble" effect. If set to False, thick lines will be drawn instead.
        :param legend: Union[str, list]
            Includes the molecule ID. If a string is given, ID will only correspond to the first molecule. List of IDs
            will correspond to the query_mol in mol[list].
        :param r_labels: tuple
            Set the number of R groups to highlight based off of the core structure.
        :param use_svg: bool
            To draw the image as SVG or PNG.
        :param source_idx_property: str
            Set the index of the atom to the source molecule. Set to "SourceAtomIdx" by default.
        :param savepath: str
            Filepath to save image. The file type will need to be specified using the "use_svg" parameter. Only .svg or
            .png images supported.
        :param return_data: bool
            Return bytes data for use in highlight_rgroups_grid()
        :param query_mol: ing
            Used for drawing a specific molecule if molecules and legend are given as a list.
        :return:
        """
        # set instance variables
        global mols
        if mol is None:
            mols = self.mol
        elif isinstance(mol, str):
            mols = [mol]
        elif isinstance(mol, list):
            mols = mol

        if core is None:
            core = self.core

        # draw molecules
        mol = [Chem.MolFromSmiles(x) for x in mols]
        core = dm.to_mol(core)

        # get coordinates
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(core)

        # match core to molecules
        ps = Chem.AdjustQueryParameters.NoAdjustments()
        ps.makeDummiesQueries = True
        core = Chem.AdjustQueryProperties(core, ps)
        mol_hydrogens = [Chem.AddHs(x, addCoords=True) for x in mol]
        mol_match = [x for x in mol_hydrogens if x.HasSubstructMatch(core)]

        # Check if any molecule matches to core structure
        if len(mol_match) != len(mol_hydrogens):
            unmatched_index = next(i for i, x in enumerate(mol_hydrogens) if not x.HasSubstructMatch(core))
            raise ValueError(f"No match to core with molecule at index {unmatched_index}. Check structure!")

        for match in mol_match:
            for atom in match.GetAtoms():
                atom.SetIntProp("SourceAtomIdx", atom.GetIdx())

        # R Group Decomposition
        rdkit.RDLogger.DisableLog('rdApp.warning')
        groups, _ = rdRGroupDecomposition.RGroupDecompose([core], mol_match, asSmiles=False, asRows=True)

        # include the atom map numbers in the substructure search in order to
        # try to ensure a good alignment of the molecule to symmetric cores
        for at in core.GetAtoms():
            if at.GetAtomMapNum():
                at.ExpandQuery(rdqueries.IsotopeEqualsQueryAtom(200 + at.GetAtomMapNum()))

        # set which molecule to query
        if query_mol is None:
            row = groups[0]
            mol = Chem.Mol(mol_match[0])
        else:
            # if a list of mols is given
            row = groups[query_mol]
            mol = Chem.Mol(mol_match[query_mol])

            # set options for mol ids
            if isinstance(legend, list):
                legend = legend[query_mol]
            elif isinstance(legend, str):
                try:
                    legend = [legend][query_mol]
                except IndexError as e:
                    if str(e) == "list index out of range":
                        raise ValueError(
                            "The legend ID is a string, but a list of query_mol is given! legend=id parameter must be given! ")
            else:
                legend = ""

        for label in row:
            if label == 'Core':
                continue
            rg = row[label]
            for at in rg.GetAtoms():
                if not at.GetAtomicNum() and at.GetAtomMapNum() and at.HasProp('dummyLabel') and at.GetProp(
                        'dummyLabel') == label:
                    # attachment point. the atoms connected to this
                    # should be from the molecule
                    for nbr in at.GetNeighbors():
                        if nbr.HasProp(source_idx_property):
                            mAt = mol.GetAtomWithIdx(nbr.GetIntProp(source_idx_property))
                            if mAt.GetIsotope():
                                mAt.SetIntProp('_OrigIsotope', mAt.GetIsotope())
                            mAt.SetIsotope(200 + at.GetAtomMapNum())
        # remove unmapped hs so that they don't mess up the depiction
        rhps = Chem.RemoveHsParameters()
        rhps.removeMapped = False
        tmol = Chem.RemoveHs(mol, rhps)
        rdDepictor.GenerateDepictionMatching2DStructure(tmol, core)

        oldNewAtomMap = {}
        # reset the original isotope values and account for the fact that
        # removing the Hs changed atom indices
        for i, at in enumerate(tmol.GetAtoms()):
            if at.HasProp(source_idx_property):
                oldNewAtomMap[at.GetIntProp(source_idx_property)] = i
                if at.HasProp("_OrigIsotope"):
                    at.SetIsotope(at.GetIntProp("_OrigIsotope"))
                    at.ClearProp("_OrigIsotope")
                else:
                    at.SetIsotope(0)

        # set colormap. Will default to three colorblind color palettes.
        # "Tol" colormap from https://davidmathlogic.com/colorblind
        tol = [(51, 34, 136), (17, 119, 51), (68, 170, 153), (136, 204, 238), (221, 204, 119), (204, 102, 119),
               (170, 68, 153), (136, 34, 85)]
        # "IBM" colormap from https://davidmathlogic.com/colorblind
        ibm = [(100, 143, 255), (120, 94, 240), (220, 38, 127), (254, 97, 0), (255, 176, 0)]
        # Okabe_Ito colormap from https://jfly.uni-koeln.de/color/
        okabe = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178),
                 (213, 94, 0), (204, 121, 167)]

        colorblind_styles = {'tol': tol, 'ibm': ibm, 'okabe': okabe}

        if colors is None:
            color_list = okabe
            for i, x in enumerate(color_list):
                color_list[i] = tuple(y / 255 for y in x)
        elif isinstance(colors, str) and colors in colorblind_styles:
            color_list = colorblind_styles[colors]
            for i, x in enumerate(color_list):
                color_list[i] = tuple(y / 255 for y in x)
        elif isinstance(colors, str) and colors not in colorblind_styles:
            raise ValueError(
                "Colors must be of 'tol', 'ibm', 'okabe' color pallette or a list of color names or hex codes")
        elif isinstance(colors, list):
            color_list = _convert_colors_to_rgb(colors)
        else:
            color_list = _convert_colors_to_rgb(colors)

        # Identify and store which atoms, bonds, and rings we'll be highlighting
        highlightatoms = defaultdict(list)
        highlightbonds = defaultdict(list)
        atomrads = {}
        widthmults = {}

        rings = []

        for i, label in enumerate(r_labels):
            color = color_list[i % len(color_list)]
            # if only looking for 1 R group, override the enumerate and use first color from color_list:
            if label == 'R':
                label = 'R1'
                color = color_list[0]
            if label == '1':
                label = 'R1'
                color = color_list[0]
            rquery = row[label]
            Chem.GetSSSR(rquery)
            rinfo = rquery.GetRingInfo()
            for at in rquery.GetAtoms():
                if at.HasProp(source_idx_property):
                    origIdx = oldNewAtomMap[at.GetIntProp(source_idx_property)]
                    highlightatoms[origIdx].append(color)
                    atomrads[origIdx] = 0.4
            if fill_ring:
                for aring in rinfo.AtomRings():
                    tring = []
                    allFound = True
                    for aid in aring:
                        at = rquery.GetAtomWithIdx(aid)
                        if not at.HasProp(source_idx_property):
                            allFound = False
                            break
                        tring.append(oldNewAtomMap[at.GetIntProp(source_idx_property)])
                    if allFound:
                        rings.append((tring, color))
            for qbnd in rquery.GetBonds():
                batom = qbnd.GetBeginAtom()
                eatom = qbnd.GetEndAtom()
                if batom.HasProp(source_idx_property) and eatom.HasProp(source_idx_property):
                    origBnd = tmol.GetBondBetweenAtoms(oldNewAtomMap[batom.GetIntProp(source_idx_property)],
                                                       oldNewAtomMap[eatom.GetIntProp(source_idx_property)])

                    # message if no source_idx_property
                    if origBnd is None:
                        print(
                            f"No bond found between atoms {oldNewAtomMap[batom.GetIntProp(source_idx_property)]} and {oldNewAtomMap[eatom.GetIntProp(source_idx_property)]}")

                    bndIdx = origBnd.GetIdx()
                    highlightbonds[bndIdx].append(color)
                    widthmults[bndIdx] = 2

        if use_svg is True:
            d2d = rdMolDraw2D.MolDraw2DSVG(width, height)
        else:
            d2d = rdMolDraw2D.MolDraw2DCairo(width, height)
        dos = d2d.drawOptions()
        dos.useBWAtomPalette()

        # if we are filling rings, go ahead and do that first so that we draw
        # the molecule on top of the filled rings
        if fill_ring and rings:
            # a hack to set the molecule scale
            d2d.DrawMoleculeWithHighlights(tmol, legend, dict(highlightatoms),
                                           dict(highlightbonds),
                                           atomrads, widthmults)
            d2d.ClearDrawing()
            conf = tmol.GetConformer()
            for (aring, color) in rings:
                ps = []
                for aidx in aring:
                    pos = Geometry.Point2D(conf.GetAtomPosition(aidx))
                    ps.append(pos)
                d2d.SetFillPolys(True)
                d2d.SetColour(color)
                d2d.DrawPolygon(ps)
            dos.clearBackground = False

        # ----------------------
        # now draw the molecule, with highlights:
        d2d.DrawMoleculeWithHighlights(tmol, legend, dict(highlightatoms), dict(highlightbonds),
                                       atomrads, widthmults)
        d2d.FinishDrawing()
        image_data = d2d.GetDrawingText()

        if savepath:
            _save_image_to_file(image_data, use_svg=use_svg, savepath=savepath)

        if return_data is True:
            return image_data
        elif use_svg is False:
            return Image(data=image_data)
        else:
            return SVG(data=image_data)

    def highlight_rgroups_grid(self, mol: list = None, core: str = None, colors: Union[str, list] = None,
                               r_labels: tuple = ('R1', 'R2', 'R3', 'R4'), fill_ring: bool = True,
                               legend: Union[str, list] = "", n_cols: int = 2, use_svg: bool = False,
                               sub_size: tuple = (250, 200), savepath: str = None):
        """
        Highlight R-Groups from a list of molecules and place them in a grid.
        :param mol: list
            Input molecules.Must be a list of smiles strings
        :param core: str
            Smiles string that will function as the core for the molecules.
        :param colors: Union[str, list]
            Set the highlight color palette. If a list is given, can be strings of color names, hex codes, or a
            combination of either. If a string is given, a colorblind palette can be used. Only three types are
            included: 'tol', 'ibm', or 'okabe'. The okabe color palette will be used by default.
        :param r_labels: tuple
            Set the number of R groups to highlight based off of the core structure.
        :param fill_ring: bool
            Fill the highlight to give the "bubble" effect. If set to False, thick lines will be drawn instead.
        :param legend: Union[str, list]
            Includes the molecule ID. If a string is given, ID will only correspond to the first molecule. List of IDs
            will correspond to the query_mol in mol[list].
       :param n_cols: int
            Set the number of columns for the grid.
        :param use_svg: bool
            To draw the image as SVG or PNG.
        :param sub_size: tuple
            Set the size of the molecule in the grid.
        :param savepath: str
            Filepath to save image. The file type will need to be specified using the "use_svg" parameter. Only .svg or
            .png images supported.
        :return:
        """
        global bio
        # set instance variable
        if mol is None:
            mol = self.mol
        elif isinstance(mol, str):
            mol = [mol]
        elif isinstance(mol, list):
            mol = mol

        if core is None:
            core = self.core

        # set the number of rows and cols to draw
        nRows = len(mol) // n_cols
        if len(mol) % n_cols:
            nRows += 1
        nCols = n_cols

        # set image size
        imgSize = (sub_size[0] * nCols, sub_size[1] * nRows)
        res = pilImage.new('RGB', imgSize)

        # loop through each molecule and legend in list
        for i, m in enumerate(mol):
            col = i % n_cols
            row = i // n_cols

            if use_svg is True:
                raise ValueError('Currently grid only supports png image format!')
                # need to place svg images into a grid and then generate a single svg file as output
                # svg = self.highlight_rgroups(mol=mol, core=core, width=sub_size[0], height=sub_size[1],
                #                              fill_ring=fill_ring, legend=id, r_labels=r_labels, use_svg=use_svg,
                #                              source_idx_property="SourceAtomIdx", return_data=True, decomp_groups=i)
            else:
                # highlight r groups
                png = self.highlight_rgroups(mol=mol, core=core, width=sub_size[0], height=sub_size[1],
                                             colors=colors, fill_ring=fill_ring, legend=legend, r_labels=r_labels,
                                             use_svg=use_svg, source_idx_property="SourceAtomIdx", return_data=True,
                                             query_mol=i)

                bio = BytesIO(png)
                img = pilImage.open(bio)
                res.paste(img, box=(col * sub_size[0], row * sub_size[1]))

        else:
            # this outputs the final image in png format.
            bio = BytesIO()
            res.save(bio, format='PNG')
            if savepath:
                _save_image_to_file(bio.getvalue(), use_svg=False, savepath=savepath)
            return Image(bio.getvalue())


class Lasso:
    def __init__(self):
        pass


class CircleGrid:
    def __init__(self):
        pass


class RDKitHighlight:
    def __init__(self):
        pass


def _convert_colors_to_rgba(color_list, bond=False, bubble=False):
    """
    Support function to convert color names or hex codes into RGB tuples for RDKit format.
    :param color_list: list
        List of colors. Can be color names or color hex codes.
    :param bond: bool
        Set the transparency of the bond highlights
    :return:
    """
    rgba_colors = []

    for color in color_list:
        try:
            if bond is not True:
                # Convert color name or hex to RGB tuple
                rgb = mcolors.to_rgb(color)
                # Add the corresponding transparency value (alpha)
                rgba = rgb + (0.2,)
                rgba_colors.append(rgba)
            else:
                # Bond highlights must be less transparent.
                # Adjusted to 0.5 instead of 0.2
                # Convert color name or hex to RGB tuple
                rgb = mcolors.to_rgb(color)
                # Add the corresponding transparency value (alpha)
                rgba = rgb + (0.5,)
                rgba_colors.append(rgba)
        except ValueError:
            print(f"Invalid color format: {color}")

    return rgba_colors


def _convert_colors_to_rgb(color_list):
    """
    Support function to convert colors or hex codes to RGB tuples for RDKit format
    :param color_list: list
        List of colors. Can be color names or color hex codes.
    :return:
    """
    rgb_colors = [mcolors.to_rgb(color) for color in color_list]

    return rgb_colors


# Define the function to save the PNG data to a file
def _save_image_to_file(image_data, use_svg, savepath):
    """
    Support function. Convert the PNG byte data to an image using Pillow
    :param image_data:
    :param savepath:
    :return:
    """
    if use_svg is False:
        img_byte_stream = BytesIO(image_data)
        img = pilImage.open(img_byte_stream)
        img.save(savepath, format='PNG', dpi=(300, 300))
    else:
        # save SVG taken from datamol's image_to_file
        with fsspec.open(savepath, "wb") as f:
            if isinstance(image_data, str):
                # in a terminal process
                f.write(image_data.encode())  # type: ignore
            else:
                # in a jupyter kernel process
                f.write(image_data.data.encode())  # type: ignore


if __name__ == "__main__":
    import doctest

    doctest.testmod()
