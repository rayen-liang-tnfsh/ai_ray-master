# -*- coding: utf-8 -*-
"""固定 seed=42 的同一組 train/test，測試各種降低測試 Cost 的方向。
不改切分方式、不動 20% 抽樣規則（委託要求）。"""
import io, sys, json
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
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
raw["date"] = raw["date"].astype(str)

# ---- 前處理（與筆記本 2 完全相同）----
df = raw.copy()
df = df.sort_values("date").reset_index(drop=True)
df["logvol"] = np.log(df["volume"])
df["prev_price"] = df["price"].shift(1)
df["prev_logvol"] = df["logvol"].shift(1)
df["dlogvol"] = df["logvol"] - df["prev_logvol"]
df["logvol_x_prev"] = df["logvol"] * df["prev_price"]
df["dlogvol_x_prev"] = df["dlogvol"] * df["prev_price"]
df["sin_m"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_m"] = np.cos(2 * np.pi * df["month"] / 12)
df["year"] = pd.to_datetime(df["date"]).dt.year
df["sin_doy"] = np.sin(2 * np.pi * pd.to_datetime(df["date"]).dt.dayofyear / 365.25)
df["cos_doy"] = np.cos(2 * np.pi * pd.to_datetime(df["date"]).dt.dayofyear / 365.25)

clean = df.dropna(subset=["volume", "temp30", "rain7", "prev_price", "prev_logvol"]).reset_index(drop=True)
log(f"clean rows = {len(clean)}")

BASE = ["prev_price", "logvol", "dlogvol", "logvol_x_prev", "dlogvol_x_prev", "sin_m", "cos_m"]

# ---- 固定切分（與筆記本完全相同）----
rng = np.random.default_rng(42)
order = rng.permutation(len(clean))
n_test = int(round(0.2 * len(clean)))
test_idx = order[:n_test]
train_idx = order[n_test:]


def prep(feats):
    X = clean[feats].to_numpy(float)
    y = clean["price"].to_numpy(float)
    mu = X[train_idx].mean(0)
    sd = X[train_idx].std(0)
    sd[sd == 0] = 1.0
    return (X[train_idx] - mu) / sd, y[train_idx], (X[test_idx] - mu) / sd, y[test_idx], mu, sd, X, y


def gd(Xtr, ytr, alpha=0.1, iters=6000):
    w = np.zeros(Xtr.shape[1])
    b = 0.0
    m = len(ytr)
    for _ in range(iters):
        e = Xtr @ w + b - ytr
        w -= alpha * (Xtr.T @ e / m)
        b -= alpha * e.mean()
    return w, b


def cost(X, y, w, b):
    return float(np.sum((X @ w + b - y) ** 2) / (2 * len(y)))


def run(name, feats, target_col="price", log_target=False, alpha=0.1, iters=6000, extra=None):
    """回傳 (訓練cost, 測試cost)；一律換算回原始價格尺度的 Cost。"""
    d = clean
    if extra is not None:
        d = extra
    y_tr_raw = d[target_col].to_numpy(float)[train_idx]
    y_te_raw = d[target_col].to_numpy(float)[test_idx]
    X = d[feats].to_numpy(float)
    mu = X[train_idx].mean(0)
    sd = X[train_idx].std(0)
    sd[sd == 0] = 1.0
    Xtr = (X[train_idx] - mu) / sd
    Xte = (X[test_idx] - mu) / sd

    if log_target:
        t_tr, t_te = np.log(y_tr_raw), np.log(y_te_raw)
    else:
        t_tr, t_te = y_tr_raw, y_te_raw

    w, b = gd(Xtr, t_tr, alpha, iters)
    p_tr = Xtr @ w + b
    p_te = Xte @ w + b
    if log_target:
        # 對數常態的期望值修正：exp(mu + s2/2)
        res_tr = t_tr - p_tr
        s2 = float(np.mean(res_tr ** 2))
        p_tr = np.exp(p_tr + s2 / 2)
        p_te = np.exp(p_te + s2 / 2)
    tr_c = float(np.sum((p_tr - y_tr_raw) ** 2) / (2 * len(y_tr_raw)))
    te_c = float(np.sum((p_te - y_te_raw) ** 2) / (2 * len(y_te_raw)))
    log(f"{name:52s} train={tr_c:8.3f}  test={te_c:8.3f}  nfeat={len(feats)}")
    return tr_c, te_c, p_te


log("=== 0. 現況（基準）===")
run("BASE 7 特徵（現況）", BASE)

log("\n=== 1. 加特徵：季節細化 / 天氣 / 趨勢 ===")
clean["logvol_sq"] = clean["logvol"] ** 2
clean["prev_price_sq"] = clean["prev_price"] ** 2
cands = {
    "BASE + temp30": BASE + ["temp30"],
    "BASE + rain7": BASE + ["rain7"],
    "BASE + temp30 + rain7": BASE + ["temp30", "rain7"],
    "BASE + sin_doy + cos_doy": BASE + ["sin_doy", "cos_doy"],
    "BASE + 日序季節（取代月季節）": ["prev_price", "logvol", "dlogvol", "logvol_x_prev", "dlogvol_x_prev", "sin_doy", "cos_doy"],
    "BASE + year": BASE + ["year"],
    "BASE + 量平方項": BASE + ["logvol_sq"],
    "BASE + 價平方項": BASE + ["prev_price_sq"],
}
clean["logvol_sq"] = clean["logvol"] ** 2
clean["prev_price_sq"] = clean["prev_price"] ** 2
for k, v in cands.items():
    run(k, v)

log("\n=== 2. 對數目標變數 ===")
run("BASE, log(price) 為目標", BASE, log_target=True)
run("BASE+temp30, log(price) 為目標", BASE + ["temp30"], log_target=True)

log("\n=== 3. 梯度下降超參數 ===")
for a in [0.05, 0.1, 0.2, 0.3, 0.5]:
    run(f"BASE alpha={a} iters=6000", BASE, alpha=a, iters=6000)
for it in [3000, 10000, 20000]:
    run(f"BASE alpha=0.1 iters={it}", BASE, alpha=0.1, iters=it)

log("\n=== 4. 交互項變化：換掉/多加 ===")
more = {
    "BASE - logvol_x_prev": [f for f in BASE if f != "logvol_x_prev"],
    "BASE - dlogvol_x_prev": [f for f in BASE if f != "dlogvol_x_prev"],
    "BASE - 兩個交互項": [f for f in BASE if "x_prev" not in f],
    "BASE + logvol_x_dlogvol": BASE + ["logvol_x_dlogvol"],
    "BASE + prev_price_sq + logvol_sq": BASE + ["prev_price_sq", "logvol_sq"],
}
clean["logvol_x_dlogvol"] = clean["logvol"] * clean["dlogvol"]
for k, v in more.items():
    run(k, v)

log("\n=== 5. 樣本加權 / 移除極端？ ===")
log("（略，涉及改動訓練資料定義，先看上面結果）")

with io.open("_exp1.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("done")
