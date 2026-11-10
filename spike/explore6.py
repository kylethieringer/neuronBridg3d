import io, h5py, requests, numpy as np, av
from PIL import Image
def decode(f, k):
    c = av.open(io.BytesIO(f["Channels"][k][()].tobytes()))
    v = np.stack([fr.to_ndarray(format="gray16le") for fr in c.decode(video=0)])
    a = f["Channels"].attrs; return v[:, :int(a["height"][0]), :int(a["width"][0])]
f = h5py.File("cache/spike/split.h5j", "r")
cdm = np.asarray(Image.open(io.BytesIO(requests.get("https://s3.amazonaws.com/janelia-flylight-color-depth/JRC2018_VNC_Unisex_40x_DS/FlyLight_Split-GAL4_Drivers/SS41075-20170623_29_A1-Split_GAL4-m-20x-vnc-JRC2018_VNC_Unisex_40x_DS-CDM_1.png").content)).convert("RGB"))
print("CDM shape", cdm.shape)
full = cdm.max(axis=2) > 20; print("row sums top90", full[:90].sum(), "bottom90", full[-90:].sum()); mask = full[90:]; maskb = full[:1119]
for k in ["Channel_0", "Channel_1", "Channel_2"]:
    mip = decode(f, k).max(axis=0); m2 = mip > np.percentile(mip, 95)
    print(k, "12bit max", mip.max(), "IoU top-crop", round((m2 & mask).sum() / (m2 | mask).sum(),3), "bottom-crop", round((m2 & maskb).sum() / (m2 | maskb).sum(),3))
swc = requests.get("https://s3.amazonaws.com/janelia-flylight-color-depth/JRC2018_VNC_Unisex_40x_DS/FlyEM_MANC_v1.2.1/SWC/10123.swc").text
open("cache/spike/10123.swc", "w").write(swc)
pts = np.array([[float(x) for x in l.split()[2:5]] for l in swc.splitlines() if l and not l.startswith("#")])
print("SWC n", len(pts), "min", pts.min(0), "max", pts.max(0))
open("cache/spike/mcfo.h5j", "wb").write(requests.get("https://s3.amazonaws.com/janelia-flylight-imagery/Gen1+MCFO/VT046828/VT046828-20171205_62_D5-f-40x-vnc-GAL4-JRC2018_VNC_Unisex-aligned_stack.h5j").content)
g = h5py.File("cache/spike/mcfo.h5j", "r"); print("MCFO", dict(g.attrs), dict(g["Channels"].attrs), list(g["Channels"]))
