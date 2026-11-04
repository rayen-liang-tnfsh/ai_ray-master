# -*- coding: utf-8 -*-
"""第四輪：專攻「相對常態」類特徵，要求同時滿足
   (a) 隨機切分 |t| > 2
   (b) 多切點時間驗證也要變好
"""
import io
import numpy as np
import pandas as pd

out = []


def log(s=""):
    out.append(str(s))


HERE = "線性回歸"
cols = {
    "日期": "date", "月份": "month", "平均批發價（元/公斤）": "price",
    "交易量（噸）": "volume", "台北近30日均溫（℃）": "temp30",
    "台北近7日雨量（毫米）": "rain7",
}
raw = pd.read_csv(f"{HERE}/data/taipei2_cabbage.csv", encoding="utf-8-sig").rename(columns=cols)[list(cols.values())]
raw["date"] = pd.to_datetime(raw["date"])

df = raw.sort_values("date").reset_index(drop=True).copy()
df["logvol"] = np.log(df["volume"])
df["prev_price"] = df["price"].shift(1)
df["prev_logvol"] = df["logvol"].shift(1)
df["dlogvol"] = df["logvol"] - df["prev_logvol"]
df["logvol_x_prev"] = df["logvol"] * df["prev_price"]
df["dlogvol_x_prev"] = df["dlogvol"] * df["prev_price"]
df["sin_m"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_m"] = np.cos(2 * np.pi * df["month"] / 12)

lp = df["logvol"].shift(1)   # 只用今天以前
pp = df["price"].shift(1)
for w in [5, 7, 14, 30, 45, 60]:
    df[f"vm{w}"] = lp.rolling(w, min_periods=w).mean()
    df[f"vs{w}"] = lp.rolling(w, min_periods=w).std()
    df[f"pm{w}"] = pp.rolling(w, min_periods=w).mean()
    df[f"ps{w}"] = pp.rolling(w, min_periods=w).std()

df["logvol_minus_vm30"] = df["logvol"] - df["vm30"]
df["logvol_minus_vm14"] = df["logvol"] - df["vm14"]
df["logvol_minus_vm7"] = df["logvol"] - df["vm7"]
df["logvol_minus_vm45"] = df["logvol"] - df["vm45"]
df["logvol_minus_vm60"] = df["logvol"] - df["vm60"]
df["logvol_z30"] = (df["logvol"] - df["vm30"]) / df["vs30"]
df["logvol_z14"] = (df["logvol"] - df["vm14"]) / df["vs14"]
df["prev_minus_pm30"] = df["prev_price"] - df["pm30"]
df["prev_z30"] = (df["prev_price"] - df["pm30"]) / df["ps30"]

clean = df.dropna(subset=["volume", "temp30", "rain7", "prev_price", "prev_logvol"]).reset_index(drop=True)
log(f"clean rows = {len(clean)}")

BASE = ["prev_price", "logvol", "dlogvol", "logvol_x_prev", "dlogvol_x_prev", "sin_m", "cos_m"]


def gd(X, y, alpha=0.1, iters=8000):
    w = np.zeros(X.shape[1])
    b = 0.0
    m = len(y)
    for _ in range(iters):
        e = X @ w + b - y
        w -= alpha * (X.T @ e / m)
        b -= alpha * e.mean()
    return w, b


def evaluate(feats, tr, te, d=clean):
    y = d["price"].to_numpy(float)
    X = d[feats].to_numpy(float)
    ok = np.isfinite(X).all(axis=1)
    tr, te = tr[ok[tr]], te[ok[te]]
    if len(tr) < 100 or len(te) < 20:
        return np.nan, np.nan
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    w, b = gd(Xtr, y[tr])
    tr_c = float(np.sum((Xtr @ w + b - y[tr]) ** 2) / (2 * len(tr)))
    te_c = float(np.sum((Xte @ w + b - y[te]) ** 2) / (2 * len(te)))
    return tr_c, te_c


CAND = {"BASE（現況）": BASE}
for f in ["logvol_minus_vm7", "logvol_minus_vm14", "logvol_minus_vm30",
          "logvol_minus_vm45", "logvol_minus_vm60", "logvol_z14", "logvol_z30",
          "prev_minus_pm30", "prev_z30"]:
    CAND[f"+ {f}"] = BASE + [f]
CAND["+ dlogvol_x_logvol_minus_vm30"] = BASE + ["logvol_minus_vm30", "dlogvol"]
CAND["+ vm30 本身（取代 logvol）"] = ["prev_price", "logvol_minus_vm30", "dlogvol",
                                 "logvol_x_prev", "dlogvol_x_prev", "sin_m", "cos_m"]

seeds = list(range(1, 61))
allres = {}
for name, f in CAND.items():
    sc = []
    for s in seeds:
        rg = np.random.default_rng(s)
        o = rg.permutation(len(clean))
        n2 = int(round(0.2 * len(clean)))
        _, te = evaluate(f, o[n2:], o[:n2])
        sc.append(te)
    allres[name] = np.array(sc)

n = len(clean)
nh = int(round(0.2 * n))
cutoffs = list(range(300, n - 153, 60))
multi = {}
for name, f in CAND.items():
    sc = []
    for start in cutoffs:
        _, te = evaluate(f, np.arange(0, start), np.arange(start, start + 153))
        if np.isfinite(te):
            sc.append(te)
    multi[name] = np.array(sc)

# seed 42
rg = np.random.default_rng(42)
order = rg.permutation(n)
nt = int(round(0.2 * n))
seed42 = {name: evaluate(f, order[nt:], order[:nt])[1] for name, f in CAND.items()}

b = allres["BASE（現況）"]
bm = multi["BASE（現況）"]
log(f"{'組合':40s} {'seed42':>8s} {'隨機平均':>9s} {'t值':>7s} {'贏':>6s} {'多切點時間':>10s} {'時間改善':>9s}")
for name in CAND:
    d = allres[name] - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se if se > 0 else 0
    wins = int(np.sum(d < 0))
    dm = multi[name].mean() - bm.mean()
    log(f"{name:40s} {seed42[name]:8.3f} {allres[name].mean():9.3f} {t:+7.2f} "
        f"{wins:3d}/{len(d)} {multi[name].mean():10.3f} {dm:+9.3f}")

with io.open("_exp4.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("done")
