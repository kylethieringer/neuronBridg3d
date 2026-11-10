import csv,sys,re
d=sys.argv[1]
f4=list(csv.DictReader(open(d+'/Supplemental_file4_DN_identification.tsv',encoding='utf-8'),delimiter='\t'))
t4={r['dn_type'] for r in f4 if r['LM_stack_id'].strip()}
lines=set(); 
for r in f4:
    for s in r['LM_stack_id'].split(','):
        s=s.strip()
        if s: lines.add(re.split(r'[- ]',s)[0])
f7=list(csv.DictReader(open(d+'/Supplemental_file7_MANC_DNs.tsv',encoding='utf-8'),delimiter='\t'))
m=[r for r in f7 if r['type'] in t4]
print('LM DN types',len(t4),'distinct lines',len(lines),'MANC DN bodyIds with LM type',len(m),'of',len(f7),'types present in MANC',len({r['type'] for r in m}))
z=list(csv.DictReader(open(sys.argv[2],encoding='utf-8'),delimiter='\t'))
zt={r['Type'] for r in z if any((r[k] or '').strip() for k in ['A','Bplus','B','C','Multiple','Unscored'])}
zA={r['Type'] for r in z if (r['A'] or '').strip()}
mz=[r for r in f7 if r['type'] in zt]; mA=[r for r in f7 if r['type'] in zA]
print('Zung types',len(zt),'in MANC',len({r['type'] for r in mz}),'bodyIds',len(mz),'| A-quality types',len(zA),'in MANC',len({r['type'] for r in mA}),'bodyIds',len(mA))
print('union types w/ MANC', len({r['type'] for r in f7 if r['type'] in (zt|t4)}), 'bodyIds', sum(1 for r in f7 if r['type'] in (zt|t4)))
