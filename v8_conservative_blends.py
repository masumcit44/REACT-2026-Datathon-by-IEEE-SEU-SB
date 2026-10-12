"""V8 conservative blends on the coherent core pool (honest LOFO)."""
import os, json, time
import numpy as np, pandas as pd
ROOT="d:/Desktop/React datathon"; OUT=os.path.join(ROOT,"v7_out"); AUD=os.path.join(ROOT,"v8_out")
os.makedirs(AUD,exist_ok=True)
import v7_common as C
from v7_stack import Head, to_prob
from sklearn.metrics import average_precision_score
CH_F3=0.5277283858077173; CH_F2=0.8298025375861702; TOL=0.001
def ap(y,p): return float(average_precision_score(np.asarray(y),np.asarray(p)))
def lofo(names,pool,y2,y3):
    M2=C.member_matrix(pool,"fold2",names,"prob"); M3=C.member_matrix(pool,"fold3",names,"prob")
    h2=Head("nnls","prob").fit(M2,y2); h3=Head("nnls","prob").fit(M3,y3)
    return ap(y3,h2.predict(M3)),ap(y2,h3.predict(M2))
def full_nnls(names,pool,y2,y3):
    M2=C.member_matrix(pool,"fold2",names,"prob"); M3=C.member_matrix(pool,"fold3",names,"prob")
    Mtr=np.vstack([M2,M3]); ytr=np.concatenate([y2,y3])
    hf=Head("nnls","prob").fit(Mtr,ytr)
    Mt=C.member_matrix(pool,"test",names,"prob")
    return hf,np.clip(to_prob(hf.predict(Mt),hf),0.0,1.0)
def main():
    _,_,y,y2,y3=C.load_core(verbose=False)
    core=["graphlgb","hgb183","lgb183","xgb183"]
    pool={m:{"fold2":C.load_pred(os.path.join(OUT,f"{m}_fold2.npy")),"fold3":C.load_pred(os.path.join(OUT,f"{m}_fold3.npy")),"test":C.load_pred(os.path.join(OUT,f"{m}_test.npy"))} for m in core}
    rows=[]
    f3,f2=lofo(core,pool,y2,y3); hf,t=full_nnls(core,pool,y2,y3)
    rows.append({"blend":"nnls_core","f2":f2,"f3":f3,"w":[round(float(x),4) for x in hf.weights],"m":float(t.mean())})
    np.save(os.path.join(AUD,"v8_blend_nnls_core_test.npy"),t)
    print(f"nnls_core f2={f2:.6f} f3={f3:.6f} mean={t.mean():.6f}",flush=True)
    M2=C.member_matrix(pool,"fold2",core,"prob"); M3=C.member_matrix(pool,"fold3",core,"prob"); Mt=C.member_matrix(pool,"test",core,"prob")
    f3e=ap(y3,M3.mean(1)); f2e=ap(y2,M2.mean(1)); te=Mt.mean(1)
    rows.append({"blend":"equal_core","f2":f2e,"f3":f3e,"w":[0.25]*4,"m":float(te.mean())})
    np.save(os.path.join(AUD,"v8_blend_equal_core_test.npy"),te)
    print(f"equal_core f2={f2e:.6f} f3={f3e:.6f} mean={te.mean():.6f}",flush=True)
    for tag,names in [("hgb_xgb",["hgb183","xgb183"]),("top3",["hgb183","xgb183","graphlgb"])]:
        a,b=lofo(names,pool,y2,y3); _,tt=full_nnls(names,pool,y2,y3)
        rows.append({"blend":tag,"f2":b,"f3":a,"w":"nnls","m":float(tt.mean())})
        np.save(os.path.join(AUD,f"v8_blend_{tag}_test.npy"),tt)
        print(f"{tag} f2={b:.6f} f3={a:.6f} mean={tt.mean():.6f}",flush=True)
    best=max(rows,key=lambda r:r["f3"])
    verdict="KEEP CHAMPION"
    if best["f3"]>CH_F3 and best["f2"]>=CH_F2-TOL: verdict="PROMOTE "+best["blend"]
    print(f"CHAMP f2={CH_F2:.6f} f3={CH_F3:.6f} best={best['blend']} f3={best['f3']:.6f} -> {verdict}",flush=True)
    with open(os.path.join(AUD,"v8_conservative_blends.json"),"w") as f: json.dump({"champ_f2":CH_F2,"champ_f3":CH_F3,"blends":rows,"best":best["blend"],"verdict":verdict},f,indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(AUD,"v8_conservative_blends.csv"),index=False)
if __name__=="__main__": main()
