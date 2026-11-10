import sys,openpyxl
from collections import Counter
ws=openpyxl.load_workbook(sys.argv[1],read_only=True).worksheets[0]
rows=[r[:8] for r in ws.iter_rows(values_only=True)]
h=rows[0]; d=[r for r in rows[1:] if r[0]]
for r in d[:5]: print(r)
withid=[r for r in d if r[1]]
print('rows',len(d),'with bodyId',len(withid),'types',len(set(r[3] for r in withid)))
print('conf',Counter(r[4] for r in withid))
print('lines',len(set(r[5] for r in withid if r[5])))
