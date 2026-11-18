import json
from pathlib import Path

import pytest

from nb3d import nbclient

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def offline(monkeypatch):
    """Serve NeuronBridge metadata URLs from local fixtures."""
    def fake_fetch_json(url):
        if url.endswith("/by_body/10123.json"):
            return json.loads((FIX / "by_body_10123.json").read_text())
        if url.endswith("/cdsresults/3333196959159296131.json"):
            return json.loads((FIX / "cds_manc_10123_top40.json").read_text())
        raise nbclient.NotFound(url)

    monkeypatch.setattr(nbclient, "fetch_json", fake_fetch_json)


def test_get_em_images_filters_dataset(offline):
    ims = nbclient.get_em_images("manc", 10123)
    assert len(ims) == 1
    em = ims[0]
    assert em.dataset == "manc"
    assert em.body_id == 10123
    assert em.id == "3333196959159296131"
    assert em.neuron_type == "hg1 MN"
    assert em.files["AlignedBodySWC"].endswith("FlyEM_MANC_v1.2.1/SWC/10123.swc")


def test_get_cds_hits_parses_fixture(offline):
    em = nbclient.get_em_images("manc", 10123)[0]
    hits = nbclient.get_cds_hits(em)
    assert len(hits) == 40
    h = hits[0]
    assert h.line == "SS00737"
    assert h.normalized_score == pytest.approx(44547.87)
    assert h.matching_pixels == 670
    assert h.mirrored is False
    assert h.library == "FlyLight Split-GAL4 Omnibus Broad"
    assert h.channel == 1
    assert h.slide_code == "20160520_22_D5"
    assert h.anatomical_area == "VNC"
    assert "CDM" in h.files


def test_body_not_found(offline):
    with pytest.raises(nbclient.BodyNotFound) as e:
        nbclient.get_em_images("manc", 99999999)
    assert e.value.dataset == "manc" and e.value.body_id == 99999999


def test_body_present_only_in_other_dataset(offline):
    # 10123 exists in NeuronBridge but not as a BANC body
    with pytest.raises(nbclient.BodyNotFound):
        nbclient.get_em_images("banc", 10123)


def test_file_url_resolves_store_prefix():
    files = {"store": "fl:open_data:vnc", "VisuallyLosslessStack": "Gen1+MCFO/VT1/x.h5j", "CDM": "a/b.png"}
    assert nbclient.file_url(files, "VisuallyLosslessStack") == \
        "https://s3.amazonaws.com/janelia-flylight-imagery/Gen1+MCFO/VT1/x.h5j"
    assert nbclient.file_url(files, "CDM") == "https://s3.amazonaws.com/janelia-flylight-color-depth/a/b.png"
    assert nbclient.file_url(files, "Gal4Expression") is None


def test_em_images_carry_anatomical_area(monkeypatch):
    """BANC bodies have a VNC and a Brain EM image; the area says which template each is in."""
    def fake(url):
        return {"results": [
            {"id": "v", "libraryName": "FlyWire_BANC_v626", "anatomicalArea": "VNC", "files": {}},
            {"id": "b", "libraryName": "FlyWire_BANC_v626", "anatomicalArea": "Brain", "files": {}},
        ]}
    monkeypatch.setattr(nbclient, "fetch_json", fake)
    assert [(e.id, e.anatomical_area) for e in nbclient.get_em_images("banc", 1)] == [("v", "VNC"), ("b", "Brain")]


def test_em_image_area_defaults_to_vnc(offline):
    assert nbclient.get_em_images("manc", 10123)[0].anatomical_area == "VNC"
