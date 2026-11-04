import numpy as np
import pandas as pd

raw = pd.read_csv('線性回歸/data/taipei2_cabbage.csv', encoding='utf-8-sig')
cols = {"日期":"date","月份":"month","平均批發價（元/公斤）":"price","交易量（噸）":"volume",
        "台北近30日均溫（℃）":"temp30","台北近7日雨量（毫米）":"rain7"}
df = raw.rename(columns=cols)[list(cols.values())]
df = df.sort_values("date").reset_index(drop=True)
df["logvol"]=np.log(df["volume"]); df["prev_price"]=df["price"].shift(1)
df["prev_logvol"]=df["logvol"].shift(1); df["dlogvol"]=df["logvol"]-df["prev_logvol"]
df["logvol_x_prev"]=df["logvol"]*df["prev_price"]; df["dlogvol_x_prev"]=df["dlogvol"]*df["prev_price"]
df["sin_m"]=np.sin(2*np.pi*df["month"]/12); df["cos_m"]=np.cos(2*np.pi*df["month"]/12)
df["year"]=pd.to_datetime(df["date"]).dt.year
df["year_c"]=df["year"]-2024
df["y2025"]=(df["year"]==2025).astype(float)
df["y2026"]=(df["year"]==2026).astype(float)
d = df.dropna(subset=["volume","temp30","rain7","prev_price","prev_logvol"]).reset_index(drop=True)
print("資料列數", len(d))

BASE=["prev_price","logvol","dlogvol","logvol_x_prev","dlogvol_x_prev","sin_m","cos_m"]

def _std(X, tr):
    mu = X[tr].mean(0)
    sd = X[tr].std(0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd

def rand42(feats):
    X=d[feats].to_numpy(float); y=d["price"].to_numpy(float)
    r=np.random.default_rng(42); o=r.permutation(len(d)); nt=int(round(0.2*len(d)))
    te,tr=o[:nt],o[nt:]
    mu,sd=_std(X,tr)
    A=np.c_[np.ones(len(tr)),(X[tr]-mu)/sd]; B=np.c_[np.ones(len(te)),(X[te]-mu)/sd]
    th,*_=np.linalg.lstsq(A,y[tr],rcond=None)
    return np.mean((B@th-y[te])**2)/2

def cv(feats, seed):
    X=d[feats].to_numpy(float); y=d["price"].to_numpy(float)
    r=np.random.default_rng(seed); idx=r.permutation(len(d)); folds=np.array_split(idx,5)
    cs=[]
    for i in range(5):
        te=folds[i]; tr=np.concatenate([folds[j] for j in range(5) if j!=i])
        mu,sd=_std(X,tr)
        A=np.c_[np.ones(len(tr)),(X[tr]-mu)/sd]; B=np.c_[np.ones(len(te)),(X[te]-mu)/sd]
        th,*_=np.linalg.lstsq(A,y[tr],rcond=None)
        cs.append(np.mean((B@th-y[te])**2)/2)
    return np.mean(cs)

def time_splits(feats):
    X=d[feats].to_numpy(float); y=d["price"].to_numpy(float)
    out=[]
    for cf in [0.5,0.6,0.7,0.8,0.9]:
        n=int(len(d)*cf); tr=np.arange(n); te=np.arange(n,len(d))
        mu,sd=_std(X,tr)
        A=np.c_[np.ones(len(tr)),(X[tr]-mu)/sd]; B=np.c_[np.ones(len(te)),(X[te]-mu)/sd]
        th,*_=np.linalg.lstsq(A,y[tr],rcond=None)
        out.append(np.mean((B@th-y[te])**2)/2)
    return out

cands = {
 "現況 7 特徵": BASE,
 "＋year_c（數值）": BASE+["year_c"],
 "＋年份啞變數 y2025,y2026": BASE+["y2025","y2026"],
}
print(f"\n{'組合':32s} {'隨機測試':>9s} {'CV(20)':>9s}  {'時間切分 50/60/70/80/90%'}")
for n,f in cands.items():
    rr=rand42(f); m=np.mean([cv(f,1000+s) for s in range(20)]); ts=time_splits(f)
    ts_str=" ".join(f"{v:6.2f}" for v in ts)
    print(f"{n:32s} {rr:9.3f} {m:9.3f}  {ts_str}")

# 配對檢定：啞變數 vs 現況
diffs=np.array([cv(BASE+["y2025","y2026"],7000+s)-cv(BASE,7000+s) for s in range(60)])
se=diffs.std(ddof=1)/np.sqrt(len(diffs))
print(f"\n啞變數 vs 現況 配對差 {diffs.mean():+.4f} ± {se:.4f}  t={diffs.mean()/se:+.2f}  勝率 {(diffs<0).mean():.0%}")
