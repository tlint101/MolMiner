# rdkit contrib
The following folder contains code taken from the [RDKit Contrib 
folder](https://github.com/rdkit/rdkit/tree/master/Contrib). The scripts in this folder are not [generally as well 
tested](https://github.com/rdkit/rdkit/issues/2279) compared to the rest of the RDKit. However, for MolMiner needs, 
they appear acceptable. 

The following are the READMEs for each associated module. 


## IFG Algorithm.

Implementation of

    An algorithm to identify functional groups in organic molecules
    Peter Ertl

    https://jcheminf.springeropen.com/articles/10.1186/s13321-017-0225-z


Authors:
    Richard Hall,
    Guillaume Godin modified function output to be more readable


    Usage:
    ```python
        # return the list of IFG (atomIds, atoms & type) for a molecule object:
        m = Chem.MolFromSmiles(smiles)
        fgs = identify_functional_groups(m)
        print fgs
    ```
    Output example:
        [IFG(atomIds=(2,), atoms='n', type='cnc'),
         IFG(atomIds=(4, 5, 6, 7), atoms='NS(=O)=O', type='cNS(c)(=O)=O'),
         IFG(atomIds=(12,), atoms='N', type='cN'),
         IFG(atomIds=(15,), atoms='n', type='cnc')]

Notes:

This implementation of Ertl paper was made by Richard Hall in summer 2017.
Cause RDKit and Peter tool can have distinct aromaticity detection behaviours, list
of functional groups in aromatic rings may differ in presence of conjugated aromatics rings.


## npscorer
RDKit-based implementation of the method described in:

Natural Product-likeness Score and Its Application for Prioritization of Compound Libraries 
Peter Ertl, Silvio Roggo, and Ansgar Schuffenhauer
Journal of Chemical Information and Modeling, 48, 68-74 (2008)
http://pubs.acs.org/doi/abs/10.1021/ci700286x

Contribution from Peter Ertl


## sasccorer
RDKit-based implementation of the method described in:

Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
Peter Ertl and Ansgar Schuffenhauer
Journal of Cheminformatics 1:8 (2009)
http://www.jcheminf.com/content/1/1/8

Contribution from Peter Ertl and Greg Landrum

