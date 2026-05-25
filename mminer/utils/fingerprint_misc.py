"""
Support functions for fingerprint
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys


# to generate a fingerprint array from a smiles string
def smiles_to_fp(smi: str = None,
                 method: str = "morgan",
                 radius: int = 2,
                 nbits: int = 2048,
                 bitvector: bool = False,
                 include_chirality: bool = False,
                 ):
    """
    Generate fingerprint from a smiles string. Different fingerprint methods can
    be used depending on the input.
    :param smi: str
        smiles string of the molecule
    :param method: str
        method to use for fingerprint generation.
    :param radius: int
        Set the radius of the circular fingerprint. Default is 2. If a different method is used, this will be
        ignored.
    :param nbits: int
        Number of bits to use for fingerprint. If using MacCSkeys, this will be ignored.
    :param bitvector: bool
        Output fingerprint results as ExplicitBitVect or as np.Array
    :param include_chirality: bool
        Include chirality in the fingerprint.
    :return: return fingerprint as ExplicitBitVect or np.Array Default returns ExplicitBitVect.
    """

    # convert smiles to RDKit mol object
    mol = Chem.MolFromSmiles(smi)

    # Set fp methods
    if method == "morgan":
        if include_chirality:
            fp_generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=radius, fpSize=nbits, includeChirality=include_chirality
            )
            fp = fp_generator.GetFingerprint(mol)
        else:
            fp_generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=radius, fpSize=nbits
            )
            fp = fp_generator.GetFingerprint(mol)

    elif method == "feature_morgan":
        fp_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=nbits,
            includeChirality=include_chirality,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
        )
        fp = fp_generator.GetFingerprint(mol)

    elif method == "atompair":
        fp_generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=nbits, includeChirality=include_chirality)
        fp = fp_generator.GetFingerprint(mol)

    elif method == "rdkit":
        fp_generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=nbits)
        fp = fp_generator.GetFingerprint(mol)

    elif method == "torsion":
        fp_generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            fpSize=nbits
        )
        fp = fp_generator.GetFingerprint(mol)

    elif method == "maccs":
        fp = MACCSkeys.GenMACCSKeys(mol)

    else:
        raise ValueError(
            f"Only 'morgan', 'feature_morgan', 'atompair', 'rdkit', 'torsion', maccs' are supported!"
        )

    if bitvector is False:
        return np.array(fp)
    elif bitvector is True:
        return fp


if __name__ == "__main__":
    import doctest

    doctest.testmod()
