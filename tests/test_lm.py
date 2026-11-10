from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nb3d import cdm25d, lm
from nb3d.em import TEMPLATE_UM, VOXEL_UM
from nb3d.nbclient import LmHit

FIX = Path(__file__).parent / "fixtures"
Z_MAX_UM = (219 - 1) * 0.7


def blank_cdm():
    return np.zeros((1119, 573, 3), dtype=np.uint8)


def test_cdm_hue_maps_to_expected_z():
    rgb = blank_cdm()
    for col, idx in ((100, 0), (200, 128), (300, 255)):
        rgb[500, col] = cdm25d.LUT[idx]
    pts = cdm25d.cdm_to_points(rgb)
    assert len(pts) == 3
    pts = pts[np.argsort(pts[:, 0])]
    slice_um = 0.7
    assert pts[0, 2] == pytest.approx(0.0, abs=slice_um)
    assert pts[1, 2] == pytest.approx(128 / 255 * Z_MAX_UM, abs=slice_um)
    assert pts[2, 2] == pytest.approx(Z_MAX_UM, abs=slice_um)
    assert pts[0, 0] == pytest.approx(100 * VOXEL_UM[0])
    assert pts[0, 1] == pytest.approx(500 * VOXEL_UM[1])


def test_cdm_brightness_does_not_change_z():
    rgb = blank_cdm()
    full = np.array(cdm25d.LUT[200], dtype=float)
    rgb[10, 10] = full.astype(np.uint8)
    rgb[10, 20] = (full * 0.4).astype(np.uint8)
    pts = cdm25d.cdm_to_points(rgb)
    assert pts[0, 2] == pytest.approx(pts[1, 2], abs=2 * 0.7)


def test_cdm_threshold_drops_background():
    rgb = blank_cdm()
    rgb[1, 1] = (15, 10, 19)  # below 20
    rgb[2, 2] = cdm25d.LUT[50]
    assert len(cdm25d.cdm_to_points(rgb, threshold=20)) == 1


def test_cdm_label_strip_cropped():
    img = np.asarray(Image.open(FIX / "cdm_sample.png").convert("RGB"))
    assert img.shape[0] == 1209
    pts = cdm25d.cdm_to_points(img)
    assert len(pts) > 100
    assert pts[:, 1].max() <= TEMPLATE_UM[1]
    assert pts[:, 2].min() >= 0 and pts[:, 2].max() <= Z_MAX_UM + 1e-6


def test_stack_to_points_finds_bright_blob():
    vol = np.zeros((40, 60, 50), dtype=np.uint16)
    vol[20:25, 30:35, 10:15] = 4000
    pts = lm.stack_to_points(vol, VOXEL_UM)
    c = pts.mean(0)
    assert c == pytest.approx([12 * VOXEL_UM[0], 32 * VOXEL_UM[1], 22 * VOXEL_UM[2]], abs=VOXEL_UM[2])


def test_stack_subsample_deterministic():
    rng = np.random.default_rng(1)
    vol = (rng.random((30, 40, 40)) > 0.5).astype(np.uint16) * 1000
    a = lm.stack_to_points(vol, VOXEL_UM, max_points=500)
    b = lm.stack_to_points(vol, VOXEL_UM, max_points=500)
    assert len(a) == 500 and np.array_equal(a, b)


def test_decode_h5j_channel_shape():
    vol = lm.decode_h5j(FIX / "stack_sample.h5j", channel=1)
    assert vol.shape == (219, 1119, 573)
    assert vol.max() > 0


def _hit(files):
    return LmHit(image_id="i", line="SS25580", library="FlyLight Split-GAL4 Omnibus Broad",
                 slide_code="s", channel=1, normalized_score=1.0, matching_pixels=1,
                 mirrored=False, files={"store": "fl:open_data:vnc", **files})


def test_load_lm_points_falls_back_to_25d(monkeypatch):
    monkeypatch.setattr(lm, "fetch", lambda url: FIX / "cdm_sample.png")
    p = lm.load_lm_points(_hit({"CDM": "x/cdm.png"}))
    assert p.source == "2.5d" and len(p.xyz) > 100


def test_load_lm_points_uses_3d_stack(monkeypatch):
    monkeypatch.setattr(lm, "fetch", lambda url: FIX / ("stack_sample.h5j" if url.endswith(".h5j") else "cdm_sample.png"))
    p = lm.load_lm_points(_hit({"CDM": "x/cdm.png", "VisuallyLosslessStack": "x/s.h5j"}))
    assert p.source == "3d" and 100 < len(p.xyz) <= 20000
    assert p.xyz[:, 0].max() <= TEMPLATE_UM[0]
