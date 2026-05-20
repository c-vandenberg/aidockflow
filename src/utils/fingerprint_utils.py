from typing import Iterable

import numpy as np
from concurrent.futures import ThreadPoolExecutor
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


def compute_fp_batch(smiles_list: list[str], max_workers: int = 16):
    """
    SMILES -> (kept_smiles, fp_uint8) for one batch.
    Returns ([], None) if nothing valid in the batch.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        pairs = [p for p in ex.map(smiles_to_morgan_fp, smiles_list) if p]

    if not pairs:
        return [], None

    batch_smiles, batch_fps = zip(*pairs)  # tuples
    fp_uint8 = fingerprints_to_numpy(list(batch_fps))  # (n, 128) uint8

    return list(batch_smiles), fp_uint8


def iter_fp_batches(smiles_iter: Iterable[str], batch_size: int = 1_000_000, max_workers: int = 16):
    """
    Stream a gz file of SMILES and yield (batch_smiles, fp_uint8) per batch.
    Skips empty/invalid batches.
    """
    buf: list[str] = []
    for s in smiles_iter:
        buf.append(s)
        if len(buf) >= batch_size:
            batch_smiles, fp_uint8 = compute_fp_batch(buf, max_workers=max_workers)
            if fp_uint8 is not None:
                yield batch_smiles, fp_uint8
            buf.clear()

    if buf:
        batch_smiles, fp_uint8 = compute_fp_batch(buf, max_workers=max_workers)
        if fp_uint8 is not None:
            yield batch_smiles, fp_uint8
