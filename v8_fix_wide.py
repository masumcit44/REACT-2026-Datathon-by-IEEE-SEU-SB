"""Recompute the CORRECTED wide-pool NNLS test vector (to_prob expit-bug fix).

Audit artifact ONLY (v8_out/): the wide pool stays rejected by the coherence
rule (legacy frame mismatch). Never writes to submission_file/.
"""
import os, json
import numpy as np
ROOT="d:/Desktop/React datathon"; OUT=os.path.join(ROOT,"v7_out"); AUD=os.path.join(ROOT,"v8_out")
import v7_common as C
from v7_stack import Head, load_pool, POOLS
def main():
    _,_,y,y2,y3=C.load_core(verbose=False)
    pool=load_pool(POOLS["wide"]); names=sorted(pool)
    M2=C.member_matrix(pool,"fold2",names,"prob"); M3=C.member_matrix(pool,"fold3",names,"prob")
    Mtr=np.vstack([M2,M3]); ytr=np.concatenate([y2,y3])
    hf=Head("nnls","prob").fit(Mtr,ytr)
    t=np.clip(hf.predict(C.member_matrix(pool,"test",names,"prob")),0.0,1.0)
    np.save(os.path.join(AUD,"v8_wide_nnls_CORRECTED_test.npy"),t)
    print(f"corrected wide nnls test mean={t.mean():.6f} min={t.min():.6f} max={t.max():.6f}",flush=True)
    print(f"weights={[round(float(x),6) for x in hf.weights]}",flush=True)
    old=C.load_pred(os.path.join(OUT,"v7_final_test_scores_wide.npy"))
    print(f"old corrupted mean={old.mean():.6f} (expit bug)",flush=True)
    with open(os.path.join(AUD,"v8_wide_correction.json"),"w") as f:
        json.dump({"members":names,"weights":[float(x) for x in hf.weights],"corrected_mean":float(t.mean()),"corrected_min":float(t.min()),"corrected_max":float(t.max()),"old_corrupted_mean":float(old.mean()),"note":"audit artifact only; wide pool stays rejected by coherence rule"},f,indent=2)
if __name__=="__main__": main()
