import openpyxl,re,sys
from collections import Counter
ws=openpyxl.load_workbook(sys.argv[1],read_only=True).worksheets[0]
rows=list(ws.iter_rows(values_only=True)); h=[str(c).replace('\n',' ') if c else '' for c in rows[0]]
print(h)
ix={k:i for i,k in enumerate(h)}
d=[r for r in rows[1:] if r[0]]
ad=[r for r in d if r[ix['Adult/ Larval']]=='Adult']
em=[r for r in ad if r[ix['EM Body IDs']]]
vnc=[r for r in ad if r[ix['VNC only']]=='Yes']
print('adult',len(ad),'withEM',len(em),'VNConly',len(vnc),'VNConly withEM',sum(1 for r in vnc if r[ix['EM Body IDs']]))
print('VNC-only with celltype',sum(1 for r in vnc if r[ix['Cell types']]))
for r in vnc[:8]: print(r[ix['Line']],r[ix['Cell types']],r[ix['EM Body IDs']],r[ix['First author(s) & year']])
ids=[x for r in em for x in re.findall(r'\d{4,}',str(r[ix['EM Body IDs']]))]
print('len dist',Counter(len(x) for x in ids))
print(Counter(r[ix['First author(s) & year']] for r in vnc).most_common(15))
manc=[r for r in em if any(len(x)<=6 for x in re.findall(r'\d{4,}',str(r[ix['EM Body IDs']])))]
print('lines with MANC-like (<=6 digit) ids',len(manc), 'ids', sum(len([x for x in re.findall(r'\d{4,}',str(r[ix['EM Body IDs']])) if len(x)<=6]) for r in manc))
print(Counter(r[ix['First author(s) & year']] for r in manc).most_common(10))
for r in manc[:3]: print(r[ix['Line']],r[ix['Cell types']],r[ix['EM Body IDs']])
