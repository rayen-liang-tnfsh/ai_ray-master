# -*- coding: utf-8 -*-
"""第二輪：用多組切分驗證候選特徵是真的有用，還是剛好切到好日子。"""
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
df["logvol_x_prev"] = df["logvol"] * df["prev_price"]
df["dlogvol_x_prev"] = df["dlogvol"] * df["prev_price"]
df["logvol_x_dlogvol"] = df["logvol"] * df["dlogvol"]
df["prev_price_sq"] = df["prev_price"] ** 2
df["logvol_sq"] = df["logvol"] ** 2
df["dlogvol_sq"] = df["dlogvol"] ** 2
df["sin_m"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_m"] = np.cos(2 * np.pi * df["month"] / 12)
doy = df["date"].dt.dayofyear
df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
df["year"] = df["date"].dt.year

# 均值回歸訊號：目前價格/量相對於「最近 30 天常態」偏離多少。
# 全部用 shift(1) 起算，只用到「今天以前」的資料，收市後一定拿得到。
for w in [7, 14, 30]:
    df[f"pm{w}"] = df["price"].shift(1).rolling(w, min_periods=w).mean()
    df[f"vm{w}"] = df["logvol"].shift(1).rolling(w, min_periods=w).mean()
df["prev_minus_pm30"] = df["prev_price"] - df["pm30"]
df["prev_div_pm30"] = df["prev_price"] / df["pm30"]
df["logvol_minus_vm30"] = df["logvol"] - df["vm30"]
df["dlogvol_minus_vm7"] = df["dlogvol"] - (df["logvol"].shift(1).rolling(7, min_periods=7).mean() - df["logvol"].shift(2).rolling(7, min_periods=7).mean())

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


def evaluate(feats, train_i, test_i, d=clean):
    y = d["price"].to_numpy(float)
    X = d[feats].to_numpy(float)
    # 這批特徵可能有 NaN（rolling），一起剔除
    ok = np.isfinite(X).all(axis=1)
    tr = train_i[ok[train_i]]
    te = test_i[ok[test_i]]
    if len(tr) < 100:
        return np.nan, np.nan
    mu = X[tr].mean(0)
    sd = X[tr].std(0)
    sd[sd == 0] = 1.0
    Xtr = (X[tr] - mu) / sd
    Xte = (X[te] - mu) / sd
    w, b = gd(Xtr, y[tr])
    te_c = float(np.sum((Xte @ w + b - y[te]) ** 2) / (2 * len(te)))
    tr_c = float(np.sum((Xtr @ w + b - y[tr]) ** 2) / (2 * len(tr)))
    return tr_c, te_c


CAND = {
    "BASE（現況）": BASE,
    "+ temp30": BASE + ["temp30"],
    "+ year": BASE + ["year"],
    "+ logvol_x_dlogvol": BASE + ["logvol_x_dlogvol"],
    "+ prev_price_sq+logvol_sq": BASE + ["prev_price_sq", "logvol_sq"],
    "+ dlogvol_sq": BASE + ["dlogvol_sq"],
    "+ sin/cos_doy": BASE + ["sin_doy", "cos_doy"],
    "+ prev_minus_pm30": BASE + ["prev_minus_pm30"],
    "+ logvol_minus_vm30": BASE + ["logvol_minus_vm30"],
    "+ dlogvol_minus_vm7": BASE + ["dlogvol_minus_vm7"],
    "+ prev_minus_pm30 + logvol_minus_vm30": BASE + ["prev_minus_pm30", "logvol_minus_vm30"],
}

log("=== A. 固定 seed=42（與筆記本同一組切分）===")
rng = np.random.default_rng(42)
order = rng.permutation(len(clean))
n_test = int(round(0.2 * len(clean)))
t42, e42 = order[n_test:], order[:n_test]
for name, f in CAND.items():
    tr, te = evaluate(f, t42, e42)
    log(f"{name:38s} train={tr:8.3f}  test={te:8.3f}")

log("\n=== B. 20 組不同隨機切分（判斷是不是真的）===")
seeds = list(range(1, 21))
res = {}
for name, f in CAND.items():
    scores = []
    for s in seeds:
        rg = np.random.default_rng(s)
        o = rg.permutation(len(clean))
        nt = int(round(0.2 * len(clean)))
        _, te = evaluate(f, o[nt:], o[:nt])
        scores.append(te)
    res[name] = scores
    log(f"{name:38s} 平均={np.mean(scores):7.3f}  中位={np.median(scores):7.3f}  "
        f"最小={np.min(scores):7.3f}  最大={np.max(scores):7.3f}  "
        f"贏BASE次數={sum(1 for a, b in zip(scores, res['BASE（現況）']) if a < b)}/20")

base_scores = np.array(res["BASE（現況）"])
log("\n=== C. 相對基準的改善量（負=更好）===")
for name, sc in res.items():
    d = np.array(sc) - base_scores
    log(f"{name:38s} 平均改善={d.mean():+7.3f}  標準誤={d.std(ddof=1)/np.sqrt(len(d)):6.3f}")

with io.open("_exp2.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("done")
