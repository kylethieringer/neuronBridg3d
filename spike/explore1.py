import csv, json, requests
B = "https://janelia-neuronbridge-data-prod.s3.us-east-1.amazonaws.com"
ver = requests.get(B + "/current.txt").text.strip(); print("version", ver)
cfg = requests.get(f"{B}/{ver}/config.json").json()
for k, s in cfg["stores"].items():
    print("STORE", k, s["label"], s["anatomicalArea"])
    for fk, p in s["prefixes"].items(): print("   ", fk, p)
rows = list(csv.DictReader(open("research/groundtruth-sources/cheong2024/supp3.csv", encoding="latin-1")))
print("supp3 cols", list(rows[0].keys()))
print(rows[0])
