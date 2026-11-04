# -*- coding: utf-8 -*-
"""第六輪：修正列數問題（rolling 用 min_periods=1，開頭退化為 expanding mean），
確保 clean 仍是 766 筆、seed=42 切分與筆記本完全相同。
驗證標準：隨機 60 組 |t|>2 且 多切點時間也要改善。
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

# 全部只用「今天以前」的資料；min_periods=1 讓開頭退化為 expanding mean，
# 這樣不會多砍任何一列，clean 維持 766 筆。
lp = df["logvol"].shift(1)
for w in [7, 14, 30]:
    df[f"vm{w}"] = lp.rolling(w, min_periods=1).mean()
df["logvol_minus_vm30"] = df["logvol"] - df["vm30"]
df["logvol_minus_vm14"] = df["logvol"] - df["vm14"]
df["dlogvol_ma7"] = df["dlogvol"].shift(1).rolling(7, min_periods=1).mean()
df["dlogvol_anom"] = df["dlogvol"] - df["dlogvol_ma7"]

clean = df.dropna(subset=["volume", "temp30", "rain7", "prev_price", "prev_logvol"]).reset_index(drop=True)
log(f"clean rows = {len(clean)} (必須是 766)")
assert len(clean) == 766, "列數改變了！切分會跟著變，不能這樣比"

BASE = ["prev_price", "logvol", "dlogvol", "logvol_x_prev", "dlogvol_x_prev", "sin_m", "cos_m"]
LV = "logvol_minus_vm30"


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


CAND = {
    "BASE（現況）": BASE,
    "+ LV(vm30)": BASE + [LV],
    "+ LV + dlogvol_anom": BASE + [LV, "dlogvol_anom"],
    "+ LV, 去掉 logvol": [f for f in BASE if f != "logvol"] + [LV],
    "+ logvol_minus_vm14": BASE + ["logvol_minus_vm14"],
    "+ LV + temp30": BASE + [LV, "temp30"],
}

# seed 42 必須重現 9.447 / 10.003
rg = np.random.default_rng(42)
order = rg.permutation(len(clean))
nt = int(round(0.2 * len(clean)))
tr42, te42 = order[nt:], order[:nt]
log("\n=== A. seed=42（BASE 應為 9.447 / 10.003）===")
s42 = {}
for name, f in CAND.items():
    tr, te = evaluate(f, tr42, te42)
    s42[name] = te
    log(f"{name:26s} train={tr:8.3f}  test={te:8.3f}")

seeds = list(range(1, 61))
n = len(clean)
allres, multi = {}, {}
cutoffs = list(range(300, n - 153, 60))
for name, f in CAND.items():
    sc = []
    for s in seeds:
        r = np.random.default_rng(s)
        o = r.permutation(n)
        n2 = int(round(0.2 * n))
        _, te = evaluate(f, o[n2:], o[:n2])
        sc.append(te)
    allres[name] = np.array(sc)
    ms = []
    for start in cutoffs:
        _, te = evaluate(f, np.arange(0, start), np.arange(start, start + 153))
        if np.isfinite(te):
            ms.append(te)
    multi[name] = np.array(ms)

b, bm = allres["BASE（現況）"], multi["BASE（現況）"]
log(f"\n{'組合':26s} {'seed42':>8s} {'隨機平均':>9s} {'t值':>7s} {'贏':>8s} {'時間平均':>9s} {'時間改善':>9s}")
for name in CAND:
    d = allres[name] - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se if se > 0 else 0
    dm = multi[name].mean() - bm.mean()
    flag = "  <== 兩項都通過" if (t < -2 and dm < 0) else ""
    log(f"{name:26s} {s42[name]:8.3f} {allres[name].mean():9.3f} {t:+7.2f} "
        f"{int(np.sum(d<0)):3d}/{len(d)} {multi[name].mean():9.3f} {dm:+9.3f}{flag}")

with io.open("_exp6.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("done")
