import io, requests, numpy as np
from PIL import Image
C = "https://s3.amazonaws.com/janelia-flylight-color-depth/JRC2018_VNC_Unisex_40x_DS/"
for lib, bid in [("FlyEM_MANC_v1.2.1", "10123"), ("FlyWire_BANC_v626", "720575941542412805"), ("FlyWire_BANC_v626", "720575941667143601")]:
    t = requests.get(f"{C}{lib}/SWC/{bid}.swc").text
    p = np.array([[float(x) for x in l.split()[2:5]] for l in t.splitlines() if l and not l.startswith("#")])
    cdm = np.asarray(Image.open(io.BytesIO(requests.get(f"{C}{lib}/{bid}-JRC2018_VNC_Unisex_40x_DS-CDM.png").content)).convert("RGB"))
    print(lib, bid, "cdm", cdm.shape, "swc range", p.min(0).round(1), p.max(0).round(1))
    for crop in (0, 90):
        m = cdm[crop:].max(2) > 20
        ys, xs = np.nonzero(m)
        print("   crop", crop, "cdm px x-range um", (xs.min()*0.461122).round(1), (xs.max()*0.461122).round(1), "y-range um", (ys.min()*0.461122).round(1), (ys.max()*0.461122).round(1))
r = requests.get("https://janelia-neuronbridge-data-prod.s3.us-east-1.amazonaws.com/?list-type=2&prefix=v3_10_0/metadata/&delimiter=/")
print(r.status_code, r.text[:1500])
