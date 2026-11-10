import pandas as pd,sys,csv
m=pd.read_feather(sys.argv[1]); print(m.shape); print(list(m.columns))
cols=[c for c in m.columns if any(k in c.lower() for k in ['type','line','gal4','manc','fafb','driver'])]
print(cols)
z={r['dn_type'] for r in csv.DictReader(open(sys.argv[2],encoding='utf-8'),delimiter='\t') if r['LM_stack_id'].strip()}
for c in cols:
    try:
        s=m[c].astype(str); hit=s.isin(z); print(c,'neurons w/ LM DN type',hit.sum(),'types',s[hit].nunique())
    except Exception as e: print(c,e)
