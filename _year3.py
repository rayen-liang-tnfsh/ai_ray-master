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
df["yc_x_prev"]=df["year_c"]*df["prev_price"]
df["yc_x_dlogvol"]=df["year_c"]*df["dlogvol"]
# 年度水位：用「該年之前的平均價」當合法特徵（不含當天）
df["prev_ma30"]=df["price"].shift(1).rolling(30).mean()
d = df.dropna(subset=["volume","temp30","rain7","prev_price","prev_logvol"]).reset_index(drop=True)

BASE=["prev_price","logvol","dlogvol","logvol_x_prev","dlogvol_x_prev","sin_m","cos_m"]
d = d.dropna(subset=["prev_ma30"]).reset_index(drop=True)

def cv(feats, seed):
    X=d[feats].to_numpy(float); y=d["price"].to_numpy(float)
    r=np.random.default_rng(seed); idx=r.permutation(len(d)); folds=np.array_split(idx,5)
    cs=[]
    for i in range(5):
        te=folds[i]; tr=np.concatenate([folds[j] for j in range(5) if j!=i])
        mu,sd=X[tr].mean(0),X[tr].std(0)
        A=np.c_[np.ones(len(tr)),(X[tr]-mu)/sd]; B=np.c_[np.ones(len(te)),(X[te]-mu)/sd]
        th,*_=np.linalg.lstsq(A,y[tr],rcond=None)
        cs.append(np.mean((B@th-y[te])**2)/2)
    return np.mean(cs)

def time_avg(feats):
    X=d[feats].to_numpy(float); y=d["price"].to_numpy(float)
    outs=[]
    for cf in [0.5,0.6,0.7,0.8,0.9]:
        n=int(len(d)*cf); tr=np.arange(n); te=np.arange(n,len(d))
        mu,sd=X[tr].mean(0),X[tr].std(0)
        A=np.c_[np.ones(len(tr)),(X[tr]-mu)/sd]; B=np.c_[np.ones(len(te)),(X[te]-mu)/sd]
        th,*_=np.linalg.lstsq(A,y[tr],rcond=None)
        outs.append(np.mean((B@th-y[te])**2)/2)
    return np.mean(outs), outs

def rand42(feats):
    X=d[feats].to_numpy(float); y=d["price"].to_numpy(float)
    r=np.random.default_rng(42); o=r.permutation(len(d)); nt=int(round(0.2*len(d)))
    te,tr=o[:nt],o[nt:]
    mu,sd=X[tr].mean(0),X[tr].std(0)
    A=np.c_[np.ones(len(tr)),(X[tr]-mu)/sd]; B=np.c_[np.ones(len(te)),(X[te]-mu)/sd]
    th,*_=np.linalg.lstsq(A,y[tr],rcond=None)
    return np.mean((B@th-y[te])**2)/2

cands = {
 "現況 7 特徵": BASE,
 "＋year_c": BASE+["year_c"],
 "＋year_c x prev_price": BASE+["year_c","yc_x_prev"],
 "＋year_c x dlogvol": BASE+["year_c","yc_x_dlogvol"],
 "＋prev_ma30（30日水位）": BASE+["prev_ma30"],
}
print(f"{'組合':30s} {'隨機測試':>9s} {'CV(20)':>9s} {'時間平均':>9s}")
for n,f in cands.items():
    rr = rand42(f); m = np.mean([cv(f,1000+s) for s in range(20)]); ta,_ = time_avg(f)
    print(f"{n:30s} {rr:9.3f} {m:9.3f} {ta:9.3f}")
