import csv,sys,re
d=sys.argv[1]
r=list(csv.DictReader(open(d+'/supp1.csv',encoding='latin-1')))
m=[x for x in r if x['LM_match_slide_code'].strip()]
lines=set(re.split(r'-',x['LM_match_slide_code'])[0] for x in m)
print('DN rows',len(r),'bodyIds w/ slide code',len(m),'types',len(set(x['type'] for x in m)),'distinct lines',len(lines))
from collections import Counter
print('line prefixes',Counter(re.match(r'[A-Za-z]+',l).group(0) if re.match(r'[A-Za-z]+',l) else l for l in lines))
print('conf',Counter(x['confidence_1_5'] for x in m))
r3=list(csv.DictReader(open(d+'/supp3.csv',encoding='latin-1')))
m3=[x for x in r3 if x['line_match'].strip()]
ls=set(l.strip() for x in m3 for l in x['line_match'].split(','))
print('MN rows',len(r3),'bodyIds w/ line',len(m3),'types',len(set(x['type'] for x in m3)),'lines',len(ls))
print(Counter(x['match_certainty(1-5)'] for x in m3)); print(Counter(x['class'] for x in m3))
print(sorted(ls)[:40])
