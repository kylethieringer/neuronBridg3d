import pandas as pd,csv,openpyxl
m=pd.read_feather('groundtruth/banc/banc_888_meta.feather')
ws=openpyxl.load_workbook('groundtruth/ehrhardt2025/table4.xlsx',read_only=True).worksheets[0]
eh={str(r[3]).strip() for r in list(ws.iter_rows(values_only=True))[1:] if r[0] and r[3]}
mn={r['type'] for r in csv.DictReader(open('groundtruth/cheong2024/supp3.csv',encoding='latin-1')) if r['line_match'].strip()}
mt=m['manc_cell_type'].astype(str)
for n,s in [('Ehrhardt',eh),('CheongMN',mn),('both',eh|mn)]:
    h=mt.isin(s); print(n,len(s),'BANC neurons',h.sum(),'types',mt[h].nunique())
