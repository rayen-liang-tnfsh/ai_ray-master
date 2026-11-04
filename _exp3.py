# -*- coding: utf-8 -*-
"""第三輪：候選組合的穩健性 + 時間切分檢查 + 逐項消融。"""
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
df["year"] = df["date"].dt.year

# 只用到「今天以前」的資訊（shift(1) 起算）
for w in [7, 14, 30]:
    df[f"vm{w}"] = df["logvol"].shift(1).rolling(w, min_periods=w).mean()
    df[f"pm{w}"] = df["price"].shift(1).rolling(w, min_periods=w).mean()
df["logvol_minus_vm30"] = df["logvol"] - df["vm30"]
df["logvol_minus_vm14"] = df["logvol"] - df["vm14"]
df["logvol_minus_vm7"] = df["logvol"] - df["vm7"]
# 量變化的「異常」程度：今天的 dlogvol 比它平常多大
df["dlogvol_mean30"] = df["dlogvol"].shift(1).rolling(30, min_periods=30).mean()
df["dlogvol_std30"] = df["dlogvol"].shift(1).rolling(30, min_periods=30).std()
df["dlogvol_z30"] = (df["dlogvol"] - df["dlogvol_mean30"]) / df["dlogvol_std30"]
df["prev_minus_pm30"] = df["prev_price"] - df["pm30"]
df["dlogvol_minus_vm7"] = df["dlogvol"] - (df["vm7"] - df["vm7"].shift(1))

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
        return np.nan, np.nan, None
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    w, b = gd(Xtr, y[tr])
    pr, pe = Xtr @ w + b, Xte @ w + b
    tr_c = float(np.sum((pr - y[tr]) ** 2) / (2 * len(tr)))
    te_c = float(np.sum((pe - y[te]) ** 2) / (2 * len(te)))
    return tr_c, te_c, pe


CAND = {
    "BASE（現況）": BASE,
    "+ year": BASE + ["year"],
    "+ temp30": BASE + ["temp30"],
    "+ logvol_minus_vm30": BASE + ["logvol_minus_vm30"],
    "+ dlogvol_z30": BASE + ["dlogvol_z30"],
    "+ logvol_minus_vm30 + dlogvol_z30": BASE + ["logvol_minus_vm30", "dlogvol_z30"],
    "+ logvol_minus_vm14 + dlogvol_z30": BASE + ["logvol_minus_vm14", "dlogvol_z30"],
    "+ logvol_minus_vm30 + temp30": BASE + ["logvol_minus_vm30", "temp30"],
    "+ logvol_minus_vm30 + dlogvol_z30 + temp30": BASE + ["logvol_minus_vm30", "dlogvol_z30", "temp30"],
}

# ---- seed 42 ----
rng = np.random.default_rng(42)
order = rng.permutation(len(clean))
nt = int(round(0.2 * len(clean)))
t42, e42 = order[nt:], order[:nt]

log("=== A. seed=42 ===")
for name, f in CAND.items():
    tr, te, _ = evaluate(f, t42, e42)
    log(f"{name:44s} train={tr:8.3f}  test={te:8.3f}")

# ---- 50 組隨機切分 ----
log("\n=== B. 50 組隨機切分 ===")
seeds = list(range(1, 51))
allres = {}
for name, f in CAND.items():
    sc = []
    for s in seeds:
        rg = np.random.default_rng(s)
        o = rg.permutation(len(clean))
        n2 = int(round(0.2 * len(clean)))
        _, te, _ = evaluate(f, o[n2:], o[:n2])
        sc.append(te)
    allres[name] = np.array(sc)
    log(f"{name:44s} 平均={np.mean(sc):7.3f}  中位={np.median(sc):7.3f}  std={np.std(sc):6.3f}")

log("\n=== C. 相對 BASE 改善（負=更好，|t|>2 才算站得住）===")
b = allres["BASE（現況）"]
for name, sc in allres.items():
    d = sc - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    wins = int(np.sum(d < 0))
    t = d.mean() / se if se > 0 else 0
    log(f"{name:44s} 平均={d.mean():+7.3f}  t={t:+6.2f}  贏={wins}/{len(d)}")

# ---- 時間切分 ----
log("\n=== D. 時間切分（前 80% 訓練、後 20% 測試）===")
n = len(clean)
nh = int(round(0.2 * n))
for name, f in CAND.items():
    tr, te, pe = evaluate(f, np.arange(n - nh), np.arange(n - nh, n))
    log(f"{name:44s} train={tr:8.3f}  test={te:8.3f}")

# ---- 擴張視窗時間切分（多個切點）----
log("\n=== E. 多切點時間驗證（每次用前 k 天訓練、後 153 天測試）===")
for name, f in CAND.items():
    sc = []
    for start in range(300, n - 153, 60):
        _, te, _ = evaluate(f, np.arange(0, start), np.arange(start, start + 153))
        if np.isfinite(te):
            sc.append(te)
    log(f"{name:44s} 平均={np.mean(sc):7.3f}  中位={np.median(sc):7.3f}  次數={len(sc)}")

with io.open("_exp3.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("done")
