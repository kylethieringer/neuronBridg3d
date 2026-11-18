from pathlib import Path

import numpy as np
import pytest

from nb3d import em
from nb3d.nbclient import EmImage
from nb3d.query import Cell

FIX = Path(__file__).parent / "fixtures"


def make_cell():
    im = EmImage(id="x", dataset="manc", body_id=10123, neuron_type=None,
                 files={"store": "fl:open_data:vnc", "AlignedBodySWC": "t/SWC/10123.swc"})
    return Cell("manc", 10123, "hg1 MN", "L", None, [im])


def test_load_em_points_units(monkeypatch):
    monkeypatch.setattr(em, "fetch", lambda url: FIX / "em_sample.swc")
    xyz = em.load_em_points(make_cell())
    assert xyz.ndim == 2 and xyz.shape[1] == 3 and len(xyz) > 1000
    # microns in JRC2018_VNC_Unisex: x 0-264.2, y 0-516.0, z 0-153.3 (spike findings)
    assert xyz[:, 0].min() >= 0 and xyz[:, 0].max() <= em.TEMPLATE_UM[0]
    assert xyz[:, 1].min() >= 0 and xyz[:, 1].max() <= em.TEMPLATE_UM[1]
    assert xyz[:, 2].min() >= 0 and xyz[:, 2].max() <= em.TEMPLATE_UM[2]
    assert xyz[0] == pytest.approx([192.146889, 174.988401, 56.671904])


def test_mirror_twice_is_identity():
    pts = np.array([[10.0, 20.0, 30.0], [200.0, 5.0, 1.0]])
    assert np.allclose(em.mirror_x(em.mirror_x(pts)), pts)


def test_mirror_moves_left_to_right():
    pts = np.array([[em.MIDLINE_X_UM - 10, 50.0, 60.0]])
    out = em.mirror_x(pts)
    assert out[0] == pytest.approx([em.MIDLINE_X_UM + 10, 50.0, 60.0])
    assert em.MIDLINE_X_UM == pytest.approx(132.12, abs=0.01)


def test_no_swc_raises(monkeypatch):
    cell = make_cell()
    cell.em_images[0].files.pop("AlignedBodySWC")
    with pytest.raises(em.NoSkeleton):
        em.load_em_points(cell)


def test_load_em_segments_joins_each_node_to_its_parent(monkeypatch, tmp_path):
    path = tmp_path / "t.swc"
    path.write_text("# test\n1 1 0 0 0 1 -1\n2 3 10 0 0 1 1\n3 3 10 5 0 1 2\n4 3 0 7 0 1 1\n")
    monkeypatch.setattr(em, "fetch", lambda url: path)
    seg = em.load_em_segments(make_cell())
    assert seg.shape == (3, 2, 3)  # root has no parent
    assert sorted(map(tuple, seg.reshape(-1, 6).tolist())) == [
        (0, 7, 0, 0, 0, 0), (10, 0, 0, 0, 0, 0), (10, 5, 0, 10, 0, 0)]  # (node, parent)


def test_load_em_segments_real_skeleton_is_connected(monkeypatch):
    monkeypatch.setattr(em, "fetch", lambda url: FIX / "em_sample.swc")
    seg = em.load_em_segments(make_cell())
    rows = [ln.split() for ln in (FIX / "em_sample.swc").read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    assert len(seg) == sum(r[6] != "-1" for r in rows)  # one segment per non-root node (this skeleton has 2 roots)


def test_brain_template_geometry():
    """JRC2018_Unisex_20x_HR, from a brain H5J header: 1210 x 566 x 174 px at 0.5189 x 0.5189 x 1.0 um."""
    b = em.BRAIN
    assert b.px == (1210, 566, 174)
    assert b.voxel_um == pytest.approx((0.5189161, 0.5189161, 1.0))
    assert b.midline_x_um == pytest.approx(1210 * 0.5189161 / 2)
    assert b.label_strip_px == 0
    assert em.VNC.px == em.TEMPLATE_PX and em.VNC.midline_x_um == pytest.approx(em.MIDLINE_X_UM)


def test_mirror_uses_template_midline():
    pts = np.array([[em.BRAIN.midline_x_um - 10, 1.0, 2.0]])
    assert em.mirror_x(pts, em.BRAIN)[0] == pytest.approx([em.BRAIN.midline_x_um + 10, 1.0, 2.0])


def _banc_cell(with_brain=True):
    ims = [EmImage(id="v", dataset="banc", body_id=5, neuron_type=None,
                   files={"store": "fl:open_data:vnc", "AlignedBodySWC": "v/SWC/5.swc"})]
    if with_brain:
        ims.append(EmImage(id="b", dataset="banc", body_id=5, neuron_type=None, anatomical_area="Brain",
                           files={"store": "fl:open_data:brain", "CDM": "JRC2018_Unisex_20x_HR/BANC/5-CDM.png"}))
    return Cell("banc", 5, "DNp26", "L", None, ims)


def test_load_em_brain_points_from_brain_cdm(monkeypatch, tmp_path):
    """NeuronBridge gives no brain-frame skeleton (the brain SWC is the VNC one), so brain points come from
    the EM neuron's brain CDM, decoded in brain geometry."""
    from PIL import Image
    from nb3d import cdm25d
    rgb = np.zeros((566, 1210, 3), dtype=np.uint8)
    rgb[200:300, 700] = cdm25d.LUT[128]
    f = tmp_path / "em_brain.png"
    Image.fromarray(rgb).save(f)
    seen = []
    monkeypatch.setattr(em, "fetch", lambda url: seen.append(url) or f)
    xyz = em.load_em_brain_points(_banc_cell())
    assert seen == ["https://s3.amazonaws.com/janelia-flylight-color-depth/JRC2018_Unisex_20x_HR/BANC/5-CDM.png"]
    assert xyz.shape == (100, 3)
    assert xyz[:, 0] == pytest.approx(700 * em.BRAIN.voxel_um[0])
    assert xyz[:, 2] == pytest.approx(128 / 255 * 173, abs=1.0)


def test_load_em_brain_points_none_without_brain_image():
    assert em.load_em_brain_points(_banc_cell(with_brain=False)) is None
