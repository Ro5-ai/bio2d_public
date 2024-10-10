from copy import deepcopy
from typing import Any, Dict, List

from rdkit import Chem
from rdkit.Chem import AllChem, Mol

# TODO: Maybe use eval?
FINGERPRINTS = {"rdkit": Chem.RDKFingerprint, "morgan": AllChem.GetMorganFingerprintAsBitVect}
FINGERPRINTS_KWARGS = {
    "rdkit": {},
    "morgan": {"radius": 2, "nBits": 2048, "useChirality": True, "useBondTypes": True},
}

def smile_to_fingerprint(
    smiles: str,
    fingerprint_type: str = "morgan",
    add_h: bool = True,
    **kwargs,
):
    """Convert smiles to RDKit fingerprint object.

    Args:
        smiles: Smiles string
        fingerprint_type: Fingerprint type to use
        do_remove_salts: Should salts be removed from smiles?
        add_h: Add hydrogens to molecule?
        kwargs: kwargs passed to fingerprint function

    Returns:
        fp: RDKit fingerprint object
    """
    mol = Chem.MolFromSmiles(smiles, True)
    fp = mol_to_fingerprint(mol, fingerprint=fingerprint_type, add_h=add_h, **kwargs)
    return fp


def mol_to_fingerprint(mol: Mol, fingerprint: str = "morgan", add_h: bool = True, **kwargs):
    """Generate a fingerprint for RDKit molecule

    Args:
        mol: RDKit molecule object
        fingerprint: fingerprint type
            morgan: requires 'radius'
        add_h: add hydrogens?
        kwargs: kwargs passed to fingerprint generator
    Returns:
        fingerprint: RDKit fingerprint obeject
    """

    assert fingerprint in FINGERPRINTS, f"Available fingerprints: {FINGERPRINTS.keys()}"
    func = FINGERPRINTS[fingerprint]
    if len(kwargs) == 0:
        func_kwargs = FINGERPRINTS_KWARGS[fingerprint]
    else:
        func_kwargs = kwargs

    mol_cp = deepcopy(mol)
    if add_h:
        mol_cp = Chem.AddHs(mol_cp)

    mol_fp = func(mol_cp, **func_kwargs)

    return mol_fp


