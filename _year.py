import numpy as np
import pandas as pd

raw = pd.read_csv('線性回歸/data/taipei2_cabbage.csv', encoding='utf-8-sig')
cols = {"日期": "date", "月份": "month", "平均批發價（元/公斤）": "price",
        "交易量（噸）": "volume", "台北近30日均溫（℃）": "temp30", "台北近7日雨量（毫米）": "rain7"}
df = raw.rename(columns=cols)[list(cols.values())]
df = df.sort_values("date").reset_index(drop=True)
df["logvol"] = np.log(df["volume"])
df["prev_price"] = df["price"].shift(1)
df["prev_logvol"] = df["logvol"].shift(1)
df["dlogvol"] = df["logvol"] - df["prev_logvol"]
df["logvol_x_prev"] = df["logvol"] * df["prev_price"]
df["dlogvol_x_prev"] = df["dlogvol"] * df["prev_price"]
df["sin_m"] = np.sin(2*np.pi*df["month"]/12)
df["cos_m"] = np.cos(2*np.pi*df["month"]/12)
df["year"] = pd.to_datetime(df["date"]).dt.year
df["year_c"] = df["year"] - 2024
df["t"] = (pd.to_datetime(df["date"]) - pd.Timestamp("2024-01-01")).dt.days / 365.25

BASE = ["prev_price","logvol","dlogvol","logvol_x_prev","dlogvol_x_prev","sin_m","cos_m"]
d = df.dropna(subset=["volume","temp30","rain7","prev_price","prev_logvol"]).reset_index(drop=True)

print("年份價格水準：")
print(d.groupby("year")["price"].agg(["count","mean","std"]).round(2).to_string())

def ev(feats, name):
    rng = np.random.default_rng(42)
    order = rng.permutation(len(d))
    nt = int(round(0.2*len(d)))
    te, tr = order[:nt], order[nt:]
    X = d[feats].to_numpy(float); y = d["price"].to_numpy(float)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    A = np.c_[np.ones(len(tr)), (X[tr]-mu)/sd]; B = np.c_[np.ones(len(te)), (X[te]-mu)/sd]
    th, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
    ctr = np.mean((A@th-y[tr])**2)/2; cte = np.mean((B@th-y[te])**2)/2
    print(f"{name:34s} train {ctr:7.3f}  test {cte:7.3f}  ({len(feats)} 特徵)")
    return cte

print("\n隨機切分 seed=42：")
ev(BASE, "現況 7 特徵")
ev(BASE+["year"], "＋year")
ev(BASE+["year_c"], "＋year_c (2024=0)")
ev(BASE+["t"], "＋t (連續時間)")

def rep_cv(feats, reps=20, k=5):
    X = d[feats].to_numpy(float); y = d["price"].to_numpy(float)
    out = []
    for s in range(reps):
        r = np.random.default_rng(1000+s)
        idx = r.permutation(len(d)); folds = np.array_split(idx, k)
        cs = []
        for i in range(k):
            te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
            mu, sd = X[tr].mean(0), X[tr].std(0)
            A = np.c_[np.ones(len(tr)), (X[tr]-mu)/sd]; B = np.c_[np.ones(len(te)), (X[te]-mu)/sd]
            th, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            cs.append(np.mean((B@th-y[te])**2)/2)
        out.append(np.mean(cs))
    return np.mean(out), np.std(out)

print("\n重複 5-fold CV：")
for n, f in [("現況 7 特徵", BASE), ("＋year", BASE+["year"]), ("＋t", BASE+["t"])]:
    m, s = rep_cv(f)
    print(f"{n:34s} CV {m:7.3f} ± {s:.3f}")

print("\n時間切分（前 80% 訓練、後 20% 測試）：")
n_cut = int(len(d)*0.8)
def time_split(feats, name):
    X = d[feats].to_numpy(float); y = d["price"].to_numpy(float)
    tr = np.arange(n_cut); te = np.arange(n_cut, len(d))
    mu, sd = X[tr].mean(0), X[tr].std(0)
    A = np.c_[np.ones(len(tr)), (X[tr]-mu)/sd]; B = np.c_[np.ones(len(te)), (X[te]-mu)/sd]
    th, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
    cte = np.mean((B@th-y[te])**2)/2
    print(f"{name:34s} test {cte:7.3f}")
time_split(BASE, "現況 7 特徵")
time_split(BASE+["year"], "＋year")
time_split(BASE+["t"], "＋t")
