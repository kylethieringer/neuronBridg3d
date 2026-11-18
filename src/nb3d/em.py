"""EM skeletons in JRC2018_VNC_Unisex template space (microns), and the two template geometries.

NeuronBridge's ``AlignedBodySWC`` is already in template microns (docs/spike-findings.md,
"EM skeletons"), so no transform is applied.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import fetch
from .nbclient import file_url
from .query import Cell

VOXEL_UM = (0.461122, 0.461122, 0.7)
TEMPLATE_PX = (573, 1119, 219)  # x, y, z
TEMPLATE_UM = tuple(n * v for n, v in zip(TEMPLATE_PX, VOXEL_UM))  # (264.2, 516.0, 153.3)
MIDLINE_X_UM = TEMPLATE_UM[0] / 2  # 132.12


@dataclass(frozen=True)
class Template:
    """An LM/EM alignment space: size in pixels (x, y, z), voxel size in microns, and the height of the
    label strip on top of its CDM PNGs."""
    name: str
    px: tuple[int, int, int]
    voxel_um: tuple[float, float, float]
    label_strip_px: int
    label_box: tuple[int, int, int, int] | None = None  # (x0, y0, x1, y1) px of a label drawn inside the CDM

    @property
    def size_um(self) -> tuple[float, float, float]:
        return tuple(n * v for n, v in zip(self.px, self.voxel_um))

    @property
    def midline_x_um(self) -> float:
        return self.size_um[0] / 2


VNC = Template("JRC2018_VNC_Unisex_40x_DS", TEMPLATE_PX, VOXEL_UM, 90)
# From a brain H5J header (image_size, voxel_size). Brain CDMs are 1210 x 566 with no label strip, but LM ones draw a
# colour bar and "Max: n/4095" text inside the image, top right (text at x 964-1201, y 36-73 on 30 CDMs).
BRAIN = Template("JRC2018_Unisex_20x_HR", (1210, 566, 174), (0.5189161, 0.5189161, 1.0), 0, (940, 0, 1210, 90))
TEMPLATES = {"VNC": VNC, "Brain": BRAIN}


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


EM_CDM_THRESHOLD = 20  # EM CDMs are rendered on pure black, so any lit pixel is the neuron


def load_em_brain_points(cell: Cell) -> np.ndarray | None:
    """(N, 3) brain-template microns from the cell's brain EM CDM, or None if it has no brain image.

    NeuronBridge's brain-image SWC is the same VNC-frame file as the VNC one, so the EM neuron's brain
    morphology is only available as this 2.5D colour-depth render (same alignment NeuronBridge searches with)."""
    from PIL import Image

    from .cdm25d import cdm_to_points  # cdm25d imports this module

    for im in cell.em_images:
        url = file_url(im.files, "CDM")
        if im.anatomical_area == "Brain" and url:
            rgb = np.asarray(Image.open(fetch(url)).convert("RGB"))
            return cdm_to_points(rgb, threshold=EM_CDM_THRESHOLD, template=BRAIN)
    return None


def mirror_x(xyz: np.ndarray, template: Template = VNC) -> np.ndarray:
    out = np.array(xyz, dtype=float, copy=True)
    out[:, 0] = 2 * template.midline_x_um - out[:, 0]
    return out
