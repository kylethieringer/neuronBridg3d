import json, collections, requests
B = "https://janelia-neuronbridge-data-prod.s3.us-east-1.amazonaws.com/v3_10_0/metadata"
mid = "3333196959159296131"
d = requests.get(f"{B}/cdsresults/{mid}.json").json()
json.dump(d, open("spike/out/cds_manc_10123.json", "w"))
res = d["results"]
print("n results", len(res), "input keys", list(d["inputImage"].keys()))
print(json.dumps(res[0], indent=1))
libs = collections.Counter(r["image"]["libraryName"] for r in res); print(libs)
keys = collections.Counter(k for r in res for k in r["image"]["files"]); print("image file keys", keys)
mkeys = collections.Counter(k for r in res for k in r["files"]); print("match file keys", mkeys)
lines = []
for r in res:
    l = r["image"]["publishedName"]
    if l not in lines: lines.append(l)
print("distinct lines", len(lines), "first 15", lines[:15])
for t in ["SS32023", "SS47161", "SS48311"]:
    print(t, lines.index(t) + 1 if t in lines else None)
