# -*- coding: utf-8 -*-
"""第八輪：函數形式。價格與供給量在經濟上常見 price ∝ 1/volume，
比 log 更貼近真實；測試倒數、平方根等形式。"""
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
df["inv_vol"] = 1.0 / df["volume"]
df["inv_sqrt_vol"] = 1.0 / np.sqrt(df["volume"])
df["sqrt_vol"] = np.sqrt(df["volume"])
df["inv_vol100"] = 100.0 / df["volume"]

df["prev_price"] = df["price"].shift(1)
df["prev_logvol"] = df["logvol"].shift(1)
df["prev_invvol"] = df["inv_vol"].shift(1)
df["dlogvol"] = df["logvol"] - df["prev_logvol"]
df["dinvvol"] = df["inv_vol"] - df["prev_invvol"]
df["logvol_x_prev"] = df["logvol"] * df["prev_price"]
df["dlogvol_x_prev"] = df["dlogvol"] * df["prev_price"]
df["invvol_x_prev"] = df["inv_vol"] * df["prev_price"]
df["dinvvol_x_prev"] = df["dinvvol"] * df["prev_price"]
df["sin_m"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_m"] = np.cos(2 * np.pi * df["month"] / 12)

lp = df["logvol"].shift(1)
df["vm30"] = lp.rolling(30, min_periods=1).mean()
df["logvol_minus_vm30"] = df["logvol"] - df["vm30"]

clean = df.dropna(subset=["volume", "temp30", "rain7", "prev_price", "prev_logvol"]).reset_index(drop=True)
assert len(clean) == 766
log(f"clean = {len(clean)}")

SEASON = ["sin_m", "cos_m"]
BASE = ["prev_price", "logvol", "dlogvol", "logvol_x_prev", "dlogvol_x_prev"] + SEASON
LV = "logvol_minus_vm30"

CAND = {
    "BASE（現況）": BASE,
    "+ LV": BASE + [LV],
    "倒數取代 logvol": ["prev_price", "inv_vol", "dlogvol", "logvol_x_prev", "dlogvol_x_prev"] + SEASON,
    "倒數全套（含交互）": ["prev_price", "inv_vol", "dinvvol", "invvol_x_prev", "dinvvol_x_prev"] + SEASON,
    "inv_sqrt 取代": ["prev_price", "inv_sqrt_vol", "dlogvol", "logvol_x_prev", "dlogvol_x_prev"] + SEASON,
    "logvol + inv_vol 都放": BASE + ["inv_vol"],
    "logvol + sqrt_vol": BASE + ["sqrt_vol"],
    "logvol + inv_vol + LV": BASE + ["inv_vol", LV],
}


def gd(X, y, iters=8000, alpha=0.1):
    w = np.zeros(X.shape[1])
    b = 0.0
    m = len(y)
    for _ in range(iters):
        e = X @ w + b - y
        w -= alpha * (X.T @ e / m)
        b -= alpha * e.mean()
    return w, b


def ev(feats, tr, te, d=clean):
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
    return (float(np.sum((Xtr @ w + b - y[tr]) ** 2) / (2 * len(tr))),
            float(np.sum((Xte @ w + b - y[te]) ** 2) / (2 * len(te))))


n = len(clean)
rg = np.random.default_rng(42)
order = rg.permutation(n)
nt = int(round(0.2 * n))
tr42, te42 = order[nt:], order[:nt]
s42 = {k: ev(f, tr42, te42) for k, f in CAND.items()}

seeds = list(range(1, 61))
allres, multi = {}, {}
cutoffs = list(range(300, n - 153, 60))
for name, f in CAND.items():
    sc = []
    for s in seeds:
        r = np.random.default_rng(s)
        o = r.permutation(n)
        n2 = int(round(0.2 * n))
        _, te = ev(f, o[n2:], o[:n2])
        sc.append(te)
    allres[name] = np.array(sc)
    ms = [ev(f, np.arange(0, st), np.arange(st, st + 153))[1] for st in cutoffs]
    multi[name] = np.array(ms)

b0 = allres["BASE（現況）"]
bm = multi["BASE（現況）"]
log(f"\n{'組合':26s} {'seed42 train':>12s} {'seed42 test':>11s} {'隨機平均':>9s} {'t值':>7s} {'贏':>8s} {'時間平均':>9s} {'時間改善':>9s}")
for name in CAND:
    d = allres[name] - b0
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se if se > 0 else 0
    dm = multi[name].mean() - bm.mean()
    flag = "  <==" if (t < -2 and dm < 0) else ""
    log(f"{name:26s} {s42[name][0]:12.3f} {s42[name][1]:11.3f} {allres[name].mean():9.3f} "
        f"{t:+7.2f} {int(np.sum(d<0)):3d}/{len(d)} {multi[name].mean():9.3f} {dm:+9.3f}{flag}")

with io.open("_exp8.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("done")
