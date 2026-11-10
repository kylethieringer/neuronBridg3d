import json, random, collections, requests
d = json.load(open("spike/out/cds_manc_10123.json"))
P = {"VisuallyLosslessStack": "https://s3.amazonaws.com/janelia-flylight-imagery/",
     "Gal4Expression": "https://s3.amazonaws.com/janelia-flylight-imagery/",
     "CDM": "https://s3.amazonaws.com/janelia-flylight-color-depth/"}
random.seed(0)
bylib = collections.defaultdict(list)
for r in d["results"]: bylib[r["image"]["libraryName"]].append(r)
for lib, rs in bylib.items():
    sample = random.sample(rs, min(20, len(rs)))
    for key in ["VisuallyLosslessStack", "Gal4Expression", "CDM"]:
        ok = 0; sizes = []; have = 0; ex = None
        for r in sample:
            p = r["image"]["files"].get(key)
            if not p: continue
            have += 1
            h = requests.head(P[key] + p)
            if h.ok:
                ok += 1; sizes.append(int(h.headers.get("Content-Length", 0)) / 1e6); ex = ex or P[key] + p
        print(f"{lib:40s} {key:22s} listed {have}/{len(sample)} reachable {ok} meanMB {sum(sizes)/max(1,len(sizes)):.1f}")
        if ex: print("     e.g.", ex)
