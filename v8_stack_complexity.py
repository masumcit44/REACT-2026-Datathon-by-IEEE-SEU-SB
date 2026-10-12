"""V8 stack complexity check: rank/logit saturate, C grid confirms no hidden gain."""
import os, json
import numpy as np, pandas as pd
ROOT="d:/Desktop/React datathon"; OUT=os.path.join(ROOT,"v7_out"); AUD=os.path.join(ROOT,"v8_out")
import v7_common as C
from v7_stack import Head
from sklearn.metrics import average_precision_score
def ap(y,p): return float(average_precision_score(np.asarray(y),np.asarray(p)))
def lofo(names,pool,y2,y3,kind="lr",mode="full",**kw):
    M2=C.member_matrix(pool,"fold2",names,mode); M3=C.member_matrix(pool,"fold3",names,mode)
    h2=Head(kind,mode,**kw).fit(M2,y2); h3=Head(kind,mode,**kw).fit(M3,y3)
    return ap(y3,h2.predict(M3)),ap(y2,h3.predict(M2))
def main():
    _,_,y,y2,y3=C.load_core(verbose=False)
    core=["graphlgb","hgb183","lgb183","xgb183"]
    pool={m:{"fold2":C.load_pred(os.path.join(OUT,f"{m}_fold2.npy")),"fold3":C.load_pred(os.path.join(OUT,f"{m}_fold3.npy")),"test":C.load_pred(os.path.join(OUT,f"{m}_test.npy"))} for m in core}
    rows=[]
    champ=(0.829803,0.527728)
    for c in [0.005,0.01,0.03,0.05,0.1,0.3,1.0]:
        a,b=lofo(core,pool,y2,y3,"lr","full",C=c)
        rows.append({"head":f"lr_C{c}","f2":b,"f3":a}); print(f"lr_C{c}: f2={b:.6f} f3={a:.6f}",flush=True)
    for a_ in [0.01,0.1,0.3,1.0,3.0,10.0]:
        x,y_=lofo(core,pool,y2,y3,"ridge","full",alpha=a_)
        rows.append({"head":f"ridge_{a_}","f2":y_,"f3":x}); print(f"ridge_{a_}: f2={y_:.6f} f3={x:.6f}",flush=True)
    x,y_=lofo(core,pool,y2,y3,"lr","rank",C=0.1); rows.append({"head":"lr_rank_C0.1","f2":y_,"f3":x})
    print(f"lr rank-only: f2={y_:.6f} f3={x:.6f}",flush=True)
    best=max(rows,key=lambda r:r["f3"])
    print(f"best={best['head']} f3={best['f3']:.6f} vs champ f3={champ[1]:.6f}",flush=True)
    with open(os.path.join(AUD,"v8_stack_complexity.json"),"w") as f: json.dump({"champ_f2":champ[0],"champ_f3":champ[1],"rows":rows,"best":best},f,indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(AUD,"v8_stack_complexity.csv"),index=False)
if __name__=="__main__": main()
