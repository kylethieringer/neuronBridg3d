import io, re, h5py, requests, numpy as np, av
from PIL import Image
src = open("spike/ref/GradientAreaGapUtils.java").read()
block = src[src.index("short[][] lut = {"):]; block = block[:block.index("};")]
lut = np.array([[int(a) for a in t] for t in re.findall(r"\{(\d+),\s*(\d+),\s*(\d+)\}", block)], float)
print("lut entries", len(lut), "first", lut[0], "last", lut[-1])
f = h5py.File("cache/spike/split.h5j", "r")
c = av.open(io.BytesIO(f["Channels"]["Channel_0"][()].tobytes()))
v = np.stack([fr.to_ndarray(format="gray16le") for fr in c.decode(video=0)])[:, :1119, :573].astype(float)
cdm = np.asarray(Image.open("cache/spike/cdm_split.png") if False else Image.open(io.BytesIO(requests.get("https://s3.amazonaws.com/janelia-flylight-color-depth/JRC2018_VNC_Unisex_40x_DS/FlyLight_Split-GAL4_Drivers/SS41075-20170623_29_A1-Split_GAL4-m-20x-vnc-JRC2018_VNC_Unisex_40x_DS-CDM_1.png").content)).convert("RGB"))[90:].astype(float)
m = cdm.max(2) > 60
rgb = cdm[m]; rgbn = rgb / rgb.max(1, keepdims=True)
lutn = lut / lut.max(1, keepdims=True)
idx = np.argmin(((rgbn[:, None, :] - lutn[None]) ** 2).sum(2), axis=1)
zc = idx / 255 * (v.shape[0] - 1)
zs = v.argmax(0)[m]
print("pixels", m.sum(), "corr", np.corrcoef(zc, zs)[0, 1], "median |dz| slices", np.median(np.abs(zc - zs)), "rev corr", np.corrcoef(v.shape[0]-1-zc, zs)[0,1])
