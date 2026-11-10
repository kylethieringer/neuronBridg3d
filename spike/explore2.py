import json, sys, requests
B = "https://janelia-neuronbridge-data-prod.s3.us-east-1.amazonaws.com/v3_10_0"
cfg = requests.get(B + "/config.json").json()
print("stores:", list(cfg["stores"].keys()))
for bid in sys.argv[1:]:
    r = requests.get(f"{B}/metadata/by_body/{bid}.json")
    print(bid, r.status_code)
    if r.ok:
        res = r.json()["results"]
        for im in res:
            print(json.dumps(im, indent=1)[:1500])
