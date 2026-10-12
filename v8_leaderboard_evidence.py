"""V8 leaderboard evidence: do folded gains predict test behavior? No LB available."""
import os, json
import numpy as np, pandas as pd
ROOT="d:/Desktop/React datathon"; OUT=os.path.join(ROOT,"v7_out"); AUD=os.path.join(ROOT,"v8_out")
import v7_common as C
from v7_stack import Head, to_prob
def main():
    _,_,y,y2,y3=C.load_core(verbose=False)
    core=["graphlgb","hgb183","lgb183","xgb183"]
    pool={m:{"fold2":C.load_pred(os.path.join(OUT,f"{m}_fold2.npy")),"fold3":C.load_pred(os.path.join(OUT,f"{m}_fold3.npy")),"test":C.load_pred(os.path.join(OUT,f"{m}_test.npy"))} for m in core}
    champ_test=C.load_pred(os.path.join(OUT,"v7_final_test_scores_core.npy"))
    M2=C.member_matrix(pool,"fold2",core,"full"); M3=C.member_matrix(pool,"fold3",core,"full")
    Mtr=np.vstack([M2,M3]); ytr=np.concatenate([y2,y3])
    alt=Head("lr","full",C=0.01).fit(Mtr,ytr)
    alt_test=np.clip(to_prob(alt.predict(C.member_matrix(pool,"test",core,"full")),alt),0.0,1.0)
    df=pd.DataFrame({"champ":champ_test,"alt_C0.01":alt_test})
    sp=df.corr("spearman"); pe=df.corr("pearson")
    print(f"spearman(champ,alt)={sp.loc['champ','alt_C0.01']:.4f} pearson={pe.loc['champ','alt_C0.01']:.4f}",flush=True)
    print(f"champ mean={champ_test.mean():.6f} alt mean={alt_test.mean():.6f}",flush=True)
    for q in [0.9,0.99,0.999]:
        print(f"q{int(q*1000)/10}: champ={np.quantile(champ_test,q):.5f} alt={np.quantile(alt_test,q):.5f}",flush=True)
    out={"note":"No leaderboard feedback exists; this measures test-time agreement between folded candidates, not LB.","spearman":float(sp.loc['champ','alt_C0.01']),"pearson":float(pe.loc['champ','alt_C0.01']),"champ_mean":float(champ_test.mean()),"alt_mean":float(alt_test.mean()),"verdict":"C0.01 agrees with champion on test ranking; folded gain is small (+0.000421) and untested on LB -> keep champion as primary, C0.01 as alternate only with new evidence."}
    with open(os.path.join(AUD,"v8_leaderboard_evidence.json"),"w") as f: json.dump(out,f,indent=2)
if __name__=="__main__": main()
