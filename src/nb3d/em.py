"""EM skeletons in JRC2018_VNC_Unisex template space (microns).

NeuronBridge's ``AlignedBodySWC`` is already in template microns (docs/spike-findings.md,
"EM skeletons"), so no transform is applied.
"""
from __future__ import annotations

import numpy as np

from .cache import fetch
from .nbclient import file_url
from .query import Cell

VOXEL_UM = (0.461122, 0.461122, 0.7)
TEMPLATE_PX = (573, 1119, 219)  # x, y, z
TEMPLATE_UM = tuple(n * v for n, v in zip(TEMPLATE_PX, VOXEL_UM))  # (264.2, 516.0, 153.3)
MIDLINE_X_UM = TEMPLATE_UM[0] / 2  # 132.12


class NoSkeleton(LookupError):
    pass


def _parse_swc(text: str) -> np.ndarray:
    rows = [ln.split() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    return np.array([[float(r[2]), float(r[3]), float(r[4])] for r in rows], dtype=float)


def _swc_text(cell: Cell) -> str:
    """The cell's skeleton, from its first EM image with an SWC."""
    for im in cell.em_images:
        url = file_url(im.files, "AlignedBodySWC")
        if url:
            return fetch(url).read_text()
    raise NoSkeleton(f"no aligned skeleton for {cell.dataset} body {cell.body_id}")


def load_em_points(cell: Cell) -> np.ndarray:
    """(N, 3) skeleton node positions in template microns."""
    return _parse_swc(_swc_text(cell))


def load_em_segments(cell: Cell) -> np.ndarray:
    """(M, 2, 3) skeleton edges (node, parent) in template microns, for drawing the cell as a connected tree."""
    rows = [ln.split() for ln in _swc_text(cell).splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    xyz = {int(r[0]): (float(r[2]), float(r[3]), float(r[4])) for r in rows}
    segs = [(xyz[int(r[0])], xyz[int(r[6])]) for r in rows if int(r[6]) in xyz]
    return np.array(segs, dtype=float).reshape(-1, 2, 3)


def mirror_x(xyz: np.ndarray) -> np.ndarray:
    out = np.array(xyz, dtype=float, copy=True)
    out[:, 0] = 2 * MIDLINE_X_UM - out[:, 0]
    return out
