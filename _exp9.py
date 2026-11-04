# -*- coding: utf-8 -*-
"""第九輪：LV 視窗掃描 + 是否還有其他穩健特徵可疊加。"""
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

lp = df["logvol"].shift(1)
for w in [10, 20, 25, 30, 35, 40, 50]:
    df[f"LV{w}"] = df["logvol"] - lp.rolling(w, min_periods=1).mean()
# 季節性供給基準：同月份歷史平均（只用過去）
df["month_mean_lv"] = (
    df.groupby("month")["logvol"].shift(1)
    .expanding().mean()
    .reset_index(level=0, drop=True)
    .sort_index()
)

clean = df.dropna(subset=["volume", "temp30", "rain7", "prev_price", "prev_logvol"]).reset_index(drop=True)
assert len(clean) == 766
log(f"clean = {len(clean)}")

BASE = ["prev_price", "logvol", "dlogvol", "logvol_x_prev", "dlogvol_x_prev", "sin_m", "cos_m"]


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


CAND = {"BASE": BASE}
for w in [10, 20, 25, 30, 35, 40, 50]:
    CAND[f"+LV{w}"] = BASE + [f"LV{w}"]

n = len(clean)
rg = np.random.default_rng(42)
order = rg.permutation(n)
nt = int(round(0.2 * n))
tr42, te42 = order[nt:], order[:nt]
s42 = {k: ev(f, tr42, te42) for k, f in CAND.items()}

seeds = list(range(1, 61))
cutoffs = list(range(300, n - 153, 60))
allres, multi = {}, {}
for name, f in CAND.items():
    sc = []
    for s in seeds:
        r = np.random.default_rng(s)
        o = r.permutation(n)
        n2 = int(round(0.2 * n))
        sc.append(ev(f, o[n2:], o[:n2])[1])
    allres[name] = np.array(sc)
    multi[name] = np.array([ev(f, np.arange(0, st), np.arange(st, st + 153))[1] for st in cutoffs])

b0, bm = allres["BASE"], multi["BASE"]
log(f"\n{'組合':10s} {'seed42':>8s} {'隨機平均':>9s} {'t值':>7s} {'贏':>8s} {'時間平均':>9s} {'時間改善':>9s}")
for name in CAND:
    d = allres[name] - b0
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se if se > 0 else 0
    dm = multi[name].mean() - bm.mean()
    log(f"{name:10s} {s42[name][1]:8.3f} {allres[name].mean():9.3f} {t:+7.2f} "
        f"{int(np.sum(d<0)):3d}/{len(d)} {multi[name].mean():9.3f} {dm:+9.3f}")

# 加總檢查
log("\n=== 能否再疊加？(LV30 + 其他) ===")
extra = {
    "+LV30 +temp30": BASE + ["LV30", "temp30"],
    "+LV30 +dlogvol_x_dlogvol": BASE + ["LV30"],
    "+LV30 取代 logvol 但保留 dlogvol": ["prev_price", "LV30", "dlogvol", "logvol_x_prev", "dlogvol_x_prev", "sin_m", "cos_m"],
    "+LV30 +logvol_x_LV30": BASE + ["LV30"],
}
for name, f in extra.items():
    sc = []
    for s in seeds:
        r = np.random.default_rng(s)
        o = r.permutation(n)
        n2 = int(round(0.2 * n))
        sc.append(ev(f, o[n2:], o[:n2])[1])
    d = np.array(sc) - b0
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se if se > 0 else 0
    log(f"  {name:36s} seed42={s42.get(name, ev(f, tr42, te42))[1]:7.3f}  "
        f"隨機={np.mean(sc):7.3f}  t={t:+6.2f}  贏={int(np.sum(d<0))}/60")

with io.open("_exp9.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("done")
