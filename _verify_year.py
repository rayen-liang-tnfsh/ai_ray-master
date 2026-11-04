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
df["is_2025"]=(df["year"]==2025).astype(float)
df["is_2026"]=(df["year"]==2026).astype(float)

base = ["prev_price","logvol","dlogvol","logvol_x_prev","dlogvol_x_prev","sin_m","cos_m"]
feats = base + ["is_2025","is_2026"]
clean = df.dropna(subset=["volume","temp30","rain7","prev_price","prev_logvol"]).reset_index(drop=True)
print("保留", len(clean), "天")

def run(features):
    X_all = clean[features].to_numpy(float); y_all = clean["price"].to_numpy(float)
    rng = np.random.default_rng(42); order = rng.permutation(len(clean))
    n_test = int(round(0.2*len(clean)))
    test_idx, train_idx = order[:n_test], order[n_test:]
    Xtr, ytr = X_all[train_idx], y_all[train_idx]
    Xte, yte = X_all[test_idx], y_all[test_idx]
    mu, sigma = Xtr.mean(0), Xtr.std(0)
    mu = np.where(sigma==0,0,mu); sigma = np.where(sigma==0,1,sigma)
    Xtr=(Xtr-mu)/sigma; Xte=(Xte-mu)/sigma
    w=np.zeros(Xtr.shape[1]); b=0.0
    for i in range(5000):
        e=Xtr@w+b-ytr
        w=w-0.1*(Xtr.T@e/len(ytr)); b=b-0.1*e.mean()
    ct=np.sum((Xtr@w+b-ytr)**2)/(2*len(ytr)); ce=np.sum((Xte@w+b-yte)**2)/(2*len(yte))
    pred=Xte@w+b
    r2=1-np.sum((yte-pred)**2)/np.sum((yte-yte.mean())**2)
    return ct,ce,r2,w,b,mu,sigma

for name, f in [("現況 7 特徵", base), ("＋年份啞變數 9 特徵", feats)]:
    ct,ce,r2,w,b,mu,sg = run(f)
    print(f"{name:20s} 訓練 {ct:7.3f}  測試 {ce:7.3f}  R2 {r2:.4f}")

ct,ce,r2,w,b,mu,sg = run(feats)
print("\n係數（z-score 尺度）:")
for n,c in zip(feats,w): print(f"  {n:16s} {c:+.4f}")
print(f"  {'b':16s} {b:+.4f}")
print("\n原始單位:")
for n,c in zip(feats,w/sg): print(f"  {n:16s} {c:+.5f}")
print(f"  {'b':16s} {b-np.sum(w*mu/sg):+.2f}")
print("\n原始單位 sigma:", dict(zip(feats, np.round(sg,4))))
print("mu:", dict(zip(feats, np.round(mu,4))))
