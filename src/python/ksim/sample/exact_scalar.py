"""ksim.exact_scalar — exact ℤ[ω] scalar arithmetic (ω = e^{iπ/4}).

NumPy mirror of tsim.core.exact_scalar. Amplitudes are kept as 4-integer-coefficient
representations a + b·ω + c·ω² + d·ω³ with a separate power-of-2 scale, so evaluation
stays exact up to the final complex cast in `_to_complex`.

Definitions
-----------
_prod4
    Product over `axis` with per-step power-of-2 normalization.

_gf2
    (…,T,P) masks × (B,P) vals → (B,…,T) parities.
"""
from __future__ import annotations

import numpy as np

UNIT_PHASES = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1],
     [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.int64)
ONE_PLUS_PHASES = UNIT_PHASES.copy(); ONE_PLUS_PHASES[:, 0] += 1
IDENTITY4 = np.array([1, 0, 0, 0], dtype=np.int64)
_W = np.exp(1j * np.pi / 4)


def _mul4(d1, d2):
    a1, b1, c1, e1 = (d1[..., i] for i in range(4))
    a2, b2, c2, e2 = (d2[..., i] for i in range(4))
    return np.stack([
        a1*a2 + b1*e2 - c1*c2 + e1*b2,
        a1*b2 + b1*a2 + c1*e2 + e1*c2,
        a1*c2 + b1*b2 + c1*a2 - e1*e2,
        a1*e2 - b1*c2 - c1*b2 + e1*a2], axis=-1)


def _reduce_pow(power, coeffs):
    red = np.all(coeffs % 2 == 0, axis=-1) & np.any(coeffs != 0, axis=-1)
    coeffs = np.where(red[..., None], coeffs // 2, coeffs)
    return power + red.astype(power.dtype), coeffs


def _prod4(vals, axis):
    vals = np.moveaxis(vals, axis, 0)
    if vals.shape[0] == 0:
        shape = vals.shape[1:]
        acc = np.broadcast_to(IDENTITY4, shape).copy()
        return acc, np.zeros(shape[:-1], dtype=np.int64)
    acc = vals[0].astype(np.int64)
    power = np.zeros(acc.shape[:-1], dtype=np.int64)
    for k in range(1, vals.shape[0]):
        acc = _mul4(acc, vals[k].astype(np.int64))
        power, acc = _reduce_pow(power, acc)
    return acc, power


def _to_complex(coeffs, power):
    z = (coeffs[..., 0] + coeffs[..., 1]*_W + coeffs[..., 2]*1j
         + coeffs[..., 3]*np.conj(_W))
    return z * np.exp2(power.astype(np.float64))


def _gf2(masks, vals):
    return (np.einsum('gtp,bp->bgt', masks.astype(np.int64),
                      vals.astype(np.int64)) % 2).astype(np.int64)
