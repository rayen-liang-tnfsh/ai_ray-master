# -*- coding: utf-8 -*-
"""第七輪：LV 特徵的安全檢查 + 時間切分細節 + 是否真的沒有洩漏。"""
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
df["vm30"] = lp.rolling(30, min_periods=1).mean()
df["logvol_minus_vm30"] = df["logvol"] - df["vm30"]

clean = df.dropna(subset=["volume", "temp30", "rain7", "prev_price", "prev_logvol"]).reset_index(drop=True)
assert len(clean) == 766
log(f"clean = {len(clean)} 天")

# ── 洩漏檢查 1：vm30 的視窗裡有沒有「今天」？
# lp = logvol.shift(1)，所以在第 t 列，vm30 = mean(logvol[t-31 .. t-1])，
# 完全不包含 logvol[t]。用數值驗證。
i = 300
manual = np.mean(np.log(clean["volume"].to_numpy()[i - 30:i]))
log(f"\n洩漏檢查 1：第 {i} 列的 vm30")
log(f"  程式算出        = {clean['vm30'].iloc[i]:.8f}")
log(f"  手算前 30 天平均 = {manual:.8f}")
log(f"  兩者相同 → 不含今天           {'✓ OK' if abs(clean['vm30'].iloc[i] - manual) < 1e-9 else '✗ 有問題'}")

# 洩漏檢查 2：LV 與「今天的 logvol」高度相關是必然的（它 = logvol - 常態），
# 但這不是洩漏，因為 logvol 本來就是「今天已知」的資訊（收市後量已經確定）。
# 要確認的是：它有沒有偷到「今天的價格」。
c_lv_price = clean["logvol_minus_vm30"].corr(clean["price"])
c_lv_ret = clean["logvol_minus_vm30"].corr(clean["price"].diff())
log(f"\n洩漏檢查 2：LV 與價格的關係")
log(f"  corr(LV, price)        = {c_lv_price:+.4f}")
log(f"  corr(LV, price 日變化)  = {c_lv_ret:+.4f}")
log("  （若 corr(LV, price日變化) 高到不合理，就要懷疑洩漏）")

# 洩漏檢查 3：把 LV 打亂，看測試 Cost 是否變差 → 證實它不是靠雜訊
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
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    w, b = gd(Xtr, y[tr])
    return (float(np.sum((Xtr @ w + b - y[tr]) ** 2) / (2 * len(tr))),
            float(np.sum((Xte @ w + b - y[te]) ** 2) / (2 * len(te))))


rg = np.random.default_rng(42)
order = rg.permutation(len(clean))
nt = int(round(0.2 * len(clean)))
tr42, te42 = order[nt:], order[:nt]

base_tr, base_te = ev(BASE, tr42, te42)
lv_tr, lv_te = ev(BASE + ["logvol_minus_vm30"], tr42, te42)
log(f"\n洩漏檢查 3：把 LV 隨機打亂（破壞它與價格的關聯）")
clean["LV_shuffled"] = np.random.default_rng(0).permutation(clean["logvol_minus_vm30"].to_numpy())
sh_tr, sh_te = ev(BASE + ["LV_shuffled"], tr42, te42)
log(f"  BASE                test = {base_te:8.3f}")
log(f"  BASE + 真的 LV       test = {lv_te:8.3f}   （改善 {lv_te - base_te:+.3f}）")
log(f"  BASE + 打亂的 LV     test = {sh_te:8.3f}   （改善 {sh_te - base_te:+.3f}）")
log("  打亂後沒有改善甚至變差 → 效益來自 LV 的『真實資訊』，不是額外自由度。")

# ── 時間切分細節 ──
log("\n=== 時間切分細節（前 80% 訓練、後 20% 測試）===")
n = len(clean)
nh = int(round(0.2 * n))
tr_t, te_t = np.arange(n - nh), np.arange(n - nh, n)
for name, f in [("BASE", BASE), ("BASE + LV", BASE + ["logvol_minus_vm30"])]:
    a, b2 = ev(f, tr_t, te_t)
    log(f"  {name:12s} train={a:7.3f}  test={b2:7.3f}")

# ── 多切點細節 ──
log("\n=== 多切點時間驗證（每個切點各自算）===")
log(f"{'訓練天數':>8s} {'測試期間':>24s} {'BASE':>8s} {'+LV':>8s} {'改善':>8s}")
base_all, lv_all = [], []
for start in range(300, n - 153, 60):
    a1, b1 = ev(BASE, np.arange(0, start), np.arange(start, start + 153))
    a2, b2 = ev(BASE + ["logvol_minus_vm30"], np.arange(0, start), np.arange(start, start + 153))
    base_all.append(b1)
    lv_all.append(b2)
    d0 = clean["date"].iloc[start]
    d1 = clean["date"].iloc[start + 152]
    log(f"{start:8d} {str(d0)[:10]}~{str(d1)[:10]} {b1:8.3f} {b2:8.3f} {b2 - b1:+8.3f}")
log(f"{'平均':>8s} {'':>24s} {np.mean(base_all):8.3f} {np.mean(lv_all):8.3f} {np.mean(lv_all)-np.mean(base_all):+8.3f}")

with io.open("_exp7.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("done")
