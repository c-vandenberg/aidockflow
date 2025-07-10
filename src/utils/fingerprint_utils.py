import numpy as np
from rdkit import Chem, DataStructs

BYTES_PER_FP = 1024 // 8 # 128

# Morgan Fingerprint with 1024 radius == 1024‑bit ECFP4 fingerprint
mfp_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def smiles_to_morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    return smiles, mfp_gen.GetFingerprint(mol)


def fingerprints_to_numpy(fps: list) -> np.ndarray:
    """
    Convert a list of RDKit ExplicitBitVect objects to a
    (N, 128) uint8 NumPy array in one go.
    """
    # 1. Serialize each fingerprint to its packed byte form (C++ code path)
    as_bytes = bytearray(BYTES_PER_FP * len(fps))
    offset = 0
    for fp in fps:
        as_bytes[offset:offset + BYTES_PER_FP] = DataStructs.BitVectToBinaryText(fp)
        offset += BYTES_PER_FP

    # 2. Re-interpret the buffer as uint8 and reshape
    arr = np.frombuffer(as_bytes, dtype=np.uint8)

    return arr.reshape(len(fps), BYTES_PER_FP)
