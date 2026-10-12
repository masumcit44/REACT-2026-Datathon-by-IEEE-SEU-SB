"""V8 LOFO ablations on the champion core pool (member + meta-family)."""
import os, json
import numpy as np, pandas as pd
ROOT="d:/Desktop/React datathon"; OUT=os.path.join(ROOT,"v7_out"); AUD=os.path.join(ROOT,"v8_out")
import v7_common as C
from v7_stack import Head
from sklearn.metrics import average_precision_score
def ap(y,p): return float(average_precision_score(np.asarray(y),np.asarray(p)))
def lofo_lr(names,pool,y2,y3,mode="full",Cval=0.1):
    M2=C.member_matrix(pool,"fold2",names,mode); M3=C.member_matrix(pool,"fold3",names,mode)
    h2=Head("lr",mode,C=Cval).fit(M2,y2); h3=Head("lr",mode,C=Cval).fit(M3,y3)
    return ap(y3,h2.predict(M3)),ap(y2,h3.predict(M2))
def main():
    _,_,y,y2,y3=C.load_core(verbose=False)
    core=["graphlgb","hgb183","lgb183","xgb183"]
    pool={m:{"fold2":C.load_pred(os.path.join(OUT,f"{m}_fold2.npy")),"fold3":C.load_pred(os.path.join(OUT,f"{m}_fold3.npy")),"test":C.load_pred(os.path.join(OUT,f"{m}_test.npy"))} for m in core}
    base_f3,base_f2=lofo_lr(core,pool,y2,y3)
    print(f"FULL core lr_C0.1/full f2={base_f2:.6f} f3={base_f3:.6f}",flush=True)
    rows=[]
    for drop in core:
        keep=[m for m in core if m!=drop]
        a,b=lofo_lr(keep,pool,y2,y3)
        rows.append({"ablation":f"drop_{drop}","kept":"+".join(keep),"f2":b,"f3":a,"d_f3":a-base_f3,"d_f2":b-base_f2})
        print(f"drop {drop}: f2={b:.6f} f3={a:.6f} d_f3={a-base_f3:+.6f}",flush=True)
    for mode in ["prob","rank"]:
        a,b=lofo_lr(core,pool,y2,y3,mode=mode)
        rows.append({"ablation":f"meta_{mode}_only","kept":"all","f2":b,"f3":a,"d_f3":a-base_f3,"d_f2":b-base_f2})
        print(f"meta {mode}: f2={b:.6f} f3={a:.6f} d_f3={a-base_f3:+.6f}",flush=True)
    for cval in [0.01,1.0,10.0]:
        a,b=lofo_lr(core,pool,y2,y3,Cval=cval)
        rows.append({"ablation":f"C_{cval}","kept":"all","f2":b,"f3":a,"d_f3":a-base_f3,"d_f2":b-base_f2})
        print(f"C={cval}: f2={b:.6f} f3={a:.6f}",flush=True)
    with open(os.path.join(AUD,"v8_lofo_ablation.json"),"w") as f: json.dump({"base_f2":base_f2,"base_f3":base_f3,"rows":rows},f,indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(AUD,"v8_lofo_ablation.csv"),index=False)
if __name__=="__main__": main()
