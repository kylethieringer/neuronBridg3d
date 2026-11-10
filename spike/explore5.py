import io, sys, h5py, requests, numpy as np, av
url = sys.argv[1]; out = sys.argv[2]
open(out, "wb").write(requests.get(url).content)
f = h5py.File(out, "r")
def show(name, obj):
    print(name, type(obj).__name__, getattr(obj, "shape", ""), dict(obj.attrs))
f.visititems(show); print("root attrs", dict(f.attrs))
ch = f["Channels"]
for k in ch:
    data = ch[k][()].tobytes()
    c = av.open(io.BytesIO(data))
    frames = [fr.to_ndarray(format="gray16le" if "16" in str(c.streams.video[0].format) else "gray") for fr in c.decode(video=0)]
    v = np.stack(frames)
    print(k, "codec", c.streams.video[0].codec_context.name, c.streams.video[0].format, "vol", v.shape, v.dtype, "max", v.max(), "p99", np.percentile(v, 99))
