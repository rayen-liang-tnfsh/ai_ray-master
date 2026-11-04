# -*- coding: utf-8 -*-
"""第五輪：a) 穩健特徵的組合 b) 近期樣本加權 c) 其他合法角度。
驗證標準：隨機切分 |t|>2 且 多切點時間也要改善。
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
    "交易量（噸） volume": "volume", "台北近30日均溫（℃）": "temp30",
    "台北近7日雨量（毫米）": "rain7",
}
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
lp = df["logvol"].shift(1)
df["vm30"] = lp.rolling(30, min_periods=30).mean()
df["logvol_minus_vm30"] = df["logvol"] - df["vm30"]
df["dlogvol_ma7"] = df["dlogvol"].shift(1).rolling(7, min_periods=7).mean()
df["dlogvol_anom"] = df["dlogvol"] - df["dlogvol_ma7"]

clean = df.dropna(subset=["volume", "temp30", "rain7", "prev_price", "prev_logvol", "vm30"]).reset_index(drop=True)
log(f"clean rows = {len(clean)}")

BASE = ["prev_price", "logvol", "dlogvol", "logvol_x_prev", "dlogvol_x_prev", "sin_m", "cos_m"]
LV = "logvol_minus_vm30"


def gd(X, y, alpha=0.1, iters=8000, sw=None):
    w = np.zeros(X.shape[1])
    b = 0.0
    m = len(y)
    if sw is None:
        for _ in range(iters):
            e = X @ w + b - y
            w -= alpha * (X.T @ e / m)
            b -= alpha * e.mean()
    else:
        sw = sw / sw.mean()
        for _ in range(iters):
            e = X @ w + b - y
            w -= alpha * (X.T @ (e * sw) / m)
            b -= alpha * float(np.mean(e * sw))
    return w, b


def evaluate(feats, tr, te, d=clean, half_life=None):
    y = d["price"].to_numpy(float)
    X = d[feats].to_numpy(float)
    ok = np.isfinite(X).all(axis=1)
    tr, te = tr[ok[tr]], te[ok[te]]
    if len(tr) < 100 or len(te) < 20:
        return np.nan, np.nan
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    sw = None
    if half_life:
        # 依「在排序資料中的位置」給權重，越新的樣本越重
        pos = np.sort(tr)
        age = pos.max() - pos
        sw = 0.5 ** (age / half_life)
        order = np.argsort(tr)
        Xtr = Xtr[order]
        ytr = y[tr][order]
        w, b = gd(Xtr, ytr, sw=sw)
        te_c = float(np.sum((Xte @ w + b - y[te]) ** 2) / (2 * len(te)))
        tr_c = float(np.sum((Xtr @ w + b - ytr) ** 2) / (2 * len(ytr)))
        return tr_c, te_c
    w, b = gd(Xtr, y[tr])
    tr_c = float(np.sum((Xtr @ w + b - y[tr]) ** 2) / (2 * len(tr)))
    te_c = float(np.sum((Xte @ w + b - y[te]) ** 2) / (2 * len(te)))
    return tr_c, te_c


CAND = {
    "BASE": (BASE, None),
    "+LV": (BASE + [LV], None),
    "+LV, 去掉 logvol": ([f for f in BASE if f != "logvol"] + [LV], None),
    "+LV +dlogvol_anom": (BASE + [LV, "dlogvol_anom"], None),
    "BASE + 半衰期365": (BASE, 365),
    "BASE + 半衰期180": (BASE, 180),
    "+LV + 半衰期365": (BASE + [LV], 365),
    "+LV + 半衰期180": (BASE + [LV], 180),
}

seeds = list(range(1, 61))
allres, multi = {}, {}
n = len(clean)
cutoffs = list(range(300, n - 153, 60))

for name, (f, hl) in CAND.items():
    sc = []
    for s in seeds:
        rg = np.random.default_rng(s)
        o = rg.permutation(n)
        n2 = int(round(0.2 * n))
        _, te = evaluate(f, o[n2:], o[:n2], half_life=hl)
        sc.append(te)
    allres[name] = np.array(sc)
    ms = []
    for start in cutoffs:
        _, te = evaluate(f, np.arange(0, start), np.arange(start, start + 153), half_life=hl)
        if np.isfinite(te):
            ms.append(te)
    multi[name] = np.array(ms)

rg = np.random.default_rng(42)
order = rg.permutation(n)
nt = int(round(0.2 * n))
s42 = {k: evaluate(v[0], order[nt:], order[:nt], half_life=v[1])[1] for k, v in CAND.items()}

b, bm = allres["BASE"], multi["BASE"]
log(f"{'組合':30s} {'seed42':>8s} {'隨機平均':>9s} {'t值':>7s} {'贏':>7s} {'時間平均':>9s} {'時間改善':>9s}")
for name in CAND:
    d = allres[name] - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se if se > 0 else 0
    dm = multi[name].mean() - bm.mean()
    log(f"{name:30s} {s42[name]:8.3f} {allres[name].mean():9.3f} {t:+7.2f} "
        f"{int(np.sum(d<0)):3d}/{len(d)} {multi[name].mean():9.3f} {dm:+9.3f}")

with io.open("_exp5.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("done")
