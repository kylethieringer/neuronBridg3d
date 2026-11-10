import json, collections, requests
B = "https://janelia-neuronbridge-data-prod.s3.us-east-1.amazonaws.com/v3_10_0/metadata"
d = json.load(open("spike/out/cds_manc_10123.json"))
lm = next(r["image"] for r in d["results"] if r["image"]["publishedName"] == "SS32023")
print("LM", lm["libraryName"], lm["id"])
e = requests.get(f"{B}/cdsresults/{lm['id']}.json").json()["results"]
print(collections.Counter(r["image"]["libraryName"] for r in e))
banc = [r["image"] for r in e if "BANC" in r["image"]["libraryName"]][:2]
for b in banc: print(json.dumps(b, indent=1))
if banc:
    bid = banc[0]["publishedName"].split(":")[-1]
    r = requests.get(f"{B}/by_body/{bid}.json"); print("by_body", bid, r.status_code, [x["libraryName"] for x in r.json()["results"]] if r.ok else "")
    sw = banc[0]["files"].get("AlignedBodySWC")
    if sw:
        t = requests.get("https://s3.amazonaws.com/janelia-flylight-color-depth/" + sw).text
        import numpy as np
        p = np.array([[float(x) for x in l.split()[2:5]] for l in t.splitlines() if l and not l.startswith("#")]); print("BANC SWC", len(p), p.min(0), p.max(0))
# by_line lookup
r = requests.get(f"{B}/by_line/SS32023.json"); print("by_line", r.status_code)
if r.ok:
    for im in r.json()["results"]:
        print("  ", im["libraryName"], im["anatomicalArea"], im.get("objective"), im.get("channel"), im["slideCode"], sorted(im["files"]))
