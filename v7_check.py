"""V7 sanity check: core alignment + legacy pool scores + optional graph artifact check."""
import os
import sys

import numpy as np

import v7_common as C

X, Xte, y, y2, y3 = C.load_core()

print("\n[check] legacy local pool (fold OOF + test)")
pool = C.load_legacy_pool()
for n in sorted(pool):
    d = pool[n]
    print(f"  {n:14s} f2={C.ap(y2, d['fold2']):.6f} f3={C.ap(y3, d['fold3']):.6f} "
          f"test_mean={d['test'].mean():.5f}")

print("\n[check] graph v3 artifacts")
gp = os.path.join(C.OUT, "graph_v3_train.npy")
if os.path.exists(gp):
    G = np.load(gp)
    Gte = np.load(os.path.join(C.OUT, "graph_v3_test.npy"))
    print(f"  graph train {G.shape} test {Gte.shape} finite={np.isfinite(G).all()}")
else:
    print("  graph_v3_train.npy not built yet")

print("\n[check] OK")