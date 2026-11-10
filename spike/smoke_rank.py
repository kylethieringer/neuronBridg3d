import sys, time
from nb3d import query, rank
ds, q = sys.argv[1], sys.argv[2]
truth = set(sys.argv[3].split(","))
g = query.resolve(ds, q)
t = time.time()
c = rank.collect_candidates(g, top_n=200)
print("cells", [c_.body_id for c_ in g.cells], "candidates", len(c), "truth NB ranks", {x.line: x.nb_rank for x in c if x.line in truth})
r = rank.rerank(g, c, progress=lambda d, n: print(f"\r{d}/{n}", end="", flush=True) if d % 20 == 0 else None)
print("\nsecs", round(time.time() - t))
print("truth new ranks", {x.line: (x.rank, x.nb_rank, round(x.score, 3), x.source) for x in r if x.line in truth})
print("top10", [(x.line, round(x.score, 3), x.nb_rank, x.source) for x in r[:10]])
