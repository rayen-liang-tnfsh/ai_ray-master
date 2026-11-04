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

BASE = ["prev_price","logvol","dlogvol","logvol_x_prev","dlogvol_x_prev","sin_m","cos_m"]
d = df.dropna(subset=["volume","temp30","rain7","prev_price","prev_logvol"]).reset_index(drop=True)

def cv_cost(feats, seed, k=5):
    X = d[feats].to_numpy(float); y = d["price"].to_numpy(float)
    r = np.random.default_rng(seed)
    idx = r.permutation(len(d)); folds = np.array_split(idx, k)
    cs = []
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        mu, sd = X[tr].mean(0), X[tr].std(0)
        A = np.c_[np.ones(len(tr)), (X[tr]-mu)/sd]; B = np.c_[np.ones(len(te)), (X[te]-mu)/sd]
        th, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        cs.append(np.mean((B@th-y[te])**2)/2)
    return np.mean(cs)

# 配對比較：同一 seed 下兩者差異
diffs = []
for s in range(60):
    a = cv_cost(BASE, 5000+s)
    b = cv_cost(BASE+["year_c"], 5000+s)
    diffs.append(b-a)
diffs = np.array(diffs)
print(f"配對差 (加年份 - 現況) 平均 {diffs.mean():+.4f}  std {diffs.std(ddof=1):.4f}  SE {diffs.std(ddof=1)/np.sqrt(len(diffs)):.4f}")
print(f"t = {diffs.mean()/(diffs.std(ddof=1)/np.sqrt(len(diffs))):+.2f}   n={len(diffs)}  seeds")
print(f"加年份較好的比例 = {(diffs<0).mean():.1%}")

# 時間切分的配對比較（固定切點、滾動多個切點）
print("\n滾動時間切分（多個切點，看『預測未來』的表現）：")
def time_eval(feats, cut_frac):
    X = d[feats].to_numpy(float); y = d["price"].to_numpy(float)
    n = int(len(d)*cut_frac)
    tr = np.arange(n); te = np.arange(n, len(d))
    mu, sd = X[tr].mean(0), X[tr].std(0)
    A = np.c_[np.ones(len(tr)), (X[tr]-mu)/sd]; B = np.c_[np.ones(len(te)), (X[te]-mu)/sd]
    th, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
    return np.mean((B@th-y[te])**2)/2
for cf in [0.5, 0.6, 0.7, 0.8, 0.9]:
    a = time_eval(BASE, cf); b = time_eval(BASE+["year_c"], cf)
    print(f"  切點 {cf:.0%}: 現況 {a:7.3f}   加年份 {b:7.3f}   差 {b-a:+7.3f}")
