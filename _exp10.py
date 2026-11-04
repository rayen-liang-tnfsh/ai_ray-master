# -*- coding: utf-8 -*-
"""第十輪：Ridge 正則化、更多 lag 特徵、大視窗 LV，全部用 100 組切分驗證。"""
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
df["prev_price2"] = df["price"].shift(2)
df["prev_logvol"] = df["logvol"].shift(1)
df["dlogvol"] = df["logvol"] - df["prev_logvol"]
df["dlogvol2"] = df["prev_logvol"] - df["logvol"].shift(2)
df["dprice"] = df["price"] - df["prev_price"]
df["logvol_x_prev"] = df["logvol"] * df["prev_price"]
df["dlogvol_x_prev"] = df["dlogvol"] * df["prev_price"]
df["sin_m"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_m"] = np.cos(2 * np.pi * df["month"] / 12)

lp = df["logvol"].shift(1)
for w in [30, 45, 60, 90]:
    df[f"LV{w}"] = df["logvol"] - lp.rolling(w, min_periods=1).mean()

clean = df.dropna(subset=["volume", "temp30", "rain7", "prev_price", "prev_logvol", "prev_price2"]).reset_index(drop=True)
log(f"clean = {len(clean)} (含 prev_price2 會少 1 列)")

BASE = ["prev_price", "logvol", "dlogvol", "logvol_x_prev", "dlogvol_x_prev", "sin_m", "cos_m"]


def gd_reg(X, y, lam=0.0, iters=8000, alpha=0.1):
    """加 L2 正則的梯度下降（b 不正則化）。"""
    w = np.zeros(X.shape[1])
    b = 0.0
    m = len(y)
    for _ in range(iters):
        e = X @ w + b - y
        w -= alpha * (X.T @ e / m + lam * w / m)
        b -= alpha * e.mean()
    return w, b


def ev(feats, tr, te, d=clean, lam=0.0):
    y = d["price"].to_numpy(float)
    X = d[feats].to_numpy(float)
    ok = np.isfinite(X).all(axis=1)
    tr, te = tr[ok[tr]], te[ok[te]]
    if len(tr) < 100 or len(te) < 20:
        return np.nan, np.nan
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    w, b = gd_reg(Xtr, y[tr], lam=lam)
    return (float(np.sum((Xtr @ w + b - y[tr]) ** 2) / (2 * len(tr))),
            float(np.sum((Xte @ w + b - y[te]) ** 2) / (2 * len(te))))


CAND = {
    "BASE": (BASE, 0.0),
    "+LV60": (BASE + ["LV60"], 0.0),
    "+LV90": (BASE + ["LV90"], 0.0),
    "+prev_price2": (BASE + ["prev_price2"], 0.0),
    "+dlogvol2": (BASE + ["dlogvol2"], 0.0),
    "BASE ridge1": (BASE, 1.0),
    "BASE ridge5": (BASE, 5.0),
    "BASE ridge20": (BASE, 20.0),
    "+LV60 ridge5": (BASE + ["LV60"], 5.0),
    "+LV60 ridge20": (BASE + ["LV60"], 20.0),
    "+LV60 +prev_price2": (BASE + ["LV60", "prev_price2"], 0.0),
    "+LV60 取代 logvol": (["prev_price", "LV60", "dlogvol", "logvol_x_prev", "dlogvol_x_prev", "sin_m", "cos_m"], 0.0),
}

n = len(clean)
seeds = list(range(1, 101))
cutoffs = list(range(300, n - 153, 60))


def multiseed(f, lam):
    sc = []
    for s in seeds:
        r = np.random.default_rng(s)
        o = r.permutation(n)
        n2 = int(round(0.2 * n))
        sc.append(ev(f, o[n2:], o[:n2], lam=lam)[1])
    return np.array(sc)


allres, multi = {}, {}
rg = np.random.default_rng(42)
order = rg.permutation(n)
nt = int(round(0.2 * n))
tr42, te42 = order[nt:], order[:nt]
s42 = {}
for name, (f, lam) in CAND.items():
    s42[name] = ev(f, tr42, te42, lam=lam)[1]
    allres[name] = multiseed(f, lam)
    multi[name] = np.array([ev(f, np.arange(0, st), np.arange(st, st + 153), lam=lam)[1] for st in cutoffs])

b0, bm = allres["BASE"], multi["BASE"]
log(f"\n{'組合':22s} {'seed42':>8s} {'隨機100組':>10s} {'t值':>7s} {'贏':>9s} {'時間平均':>9s} {'時間改善':>9s}")
for name in CAND:
    d = allres[name] - b0
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se if se > 0 else 0
    dm = multi[name].mean() - bm.mean()
    flag = "  <==" if (t < -2 and dm < 0) else ""
    log(f"{name:22s} {s42[name]:8.3f} {allres[name].mean():10.3f} {t:+7.2f} "
        f"{int(np.sum(d<0)):4d}/{len(d)} {multi[name].mean():9.3f} {dm:+9.3f}{flag}")

with io.open("_exp10.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("done")
