# 3 訓練：梯度下降
# 三個函式各管一件事：compute_cost 算錯多少、compute_gradient 算往哪邊改、
# gradient_descent 重複走直到碗底。特徵已縮放，所有參數在同一把尺上，可共用 alpha。
import matplotlib.pyplot as plt

# ══ 3.1 Cost ═════════════════════════════════════════════
# J(w,b) = 1/(2m) × Σ (Xw + b − y)²
#   平方    ：誤差有正有負，直接相加會抵銷，平方後全部變正
#   除以 2m ：除以 m 是取平均（資料多寡不影響可比性）；除以 2 純為微分時消掉 2x 的 2
# 用平方而非絕對值：處處可微好寫梯度，代價是極端值主導整個 Cost。
def compute_cost(X, y, w, b):
    m = X.shape[0]
    error = X @ w + b - y        # 每一天的「預測 − 實際」
    return float(np.sum(error ** 2) / (2 * m))

# ══ 3.2 梯度 ═════════════════════════════════════════════
# ∂J/∂w = Xᵀ(Xw+b−y)/m      ∂J/∂b = Σ(Xw+b−y)/m
# X.T @ error 展開後第 j 個分量 = Σ(誤差ᵢ × 特徵ᵢⱼ)，
# 也就是「該特徵和誤差有多同向」——一個矩陣乘法算完所有特徵的梯度。
def compute_gradient(X, y, w, b):
    m = X.shape[0]
    error = X @ w + b - y
    return X.T @ error / m, float(np.sum(error) / m)

# ══ 3.3 梯度下降 ═════════════════════════════════════════
# w、b 必須「同時」更新：先用舊參數把兩個梯度都算完，才一起改。
# 線性回歸的 Cost 是凸函數（碗形唯一最低點），起點不影響結果。
# alpha=0.1：0.01 太慢、0.3/0.5 前期震盪大；有縮放才能共用同一個 alpha。
# 5000 步遠超過需要（約 500 步就收斂），多跑不會有害，只是浪費時間。
def gradient_descent(X, y, alpha, num_iters):
    w = np.zeros(X.shape[1])
    b = 0.0
    history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        history.append(compute_cost(X, y, w, b))
        if i % 500 == 0:
            print(f"Epoch {i:5d}: Cost {history[-1]:8.3f}")
    return w, b, history

alpha, num_iters = 0.1, 5000
w, b, history = gradient_descent(X_train, y_train, alpha, num_iters)

# ══ 3.4 訓練 / 測試 Cost ═════════════════════════════════
# 訓練 Cost 低不代表有用（模型看過答案）；測試 Cost 才是「沒見過的日子有多準」。
cost_train = compute_cost(X_train, y_train, w, b)
cost_test = compute_cost(X_test, y_test, w, b)

print("\n學到的係數（特徵已 z-score；解讀是「該特徵 +1 個標準差，價格變動幾元」）")
for name, coef in zip(features, w):
    print(f"  {name:16s} {coef:+.4f}")
print(f"  {'b':16s} {b:+.4f}")

print(f"\n訓練 Cost = {cost_train:.3f}")
print(f"測試 Cost = {cost_test:.3f}")
# Cost 單位是「元²/公斤²」，開根號 ×2 之後變回元/公斤，可以直接跟均價比。
print("Cost 開根號約", round(np.sqrt(2 * cost_test), 2), "元/公斤，是平均預測誤差的量級")

# ══ 3.5 係數換回原始單位 ═════════════════════════════════
# 縮放學到的是「x 每變 1 個標準差」的效果，要講原始單位必須換回：
#   w_原 = w / sigma，b_原 = b − Σ(w·mu/sigma)
w_orig = w / sigma
b_orig = b - np.sum(w * mu / sigma)
print("\n換回原始單位的係數")
for name, coef in zip(features, w_orig):
    print(f"  {name:16s} {coef:+.5f}")
print(f"  {'b':16s} {b_orig:+.2f}")

# ══ 3.6 兩張圖 ══════════════════════════════════════════
# 左：Cost 應該先掉很快再變平（變平 = 走到碗底）。
#     沒變平 → 步數不夠；往上翹或鋸齒亂跳 → alpha 太大。
# 右：每一點是一天。完美模型所有點都落在黑色虛線（y=x）上。
#     點在上方 = 高估，在下方 = 低估。
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(history)
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("cost J(w, b)")
axes[0].set_title("Cost falls then flattens")

pred_train = X_train @ w + b
pred_test = X_test @ w + b
axes[1].scatter(y_train, pred_train, marker="x", c="r", label="train", alpha=0.6)
axes[1].scatter(y_test, pred_test, marker="o", facecolors="none", edgecolors="b",
                s=28, label="test")
lims = [y_all.min() - 2, y_all.max() + 2]
axes[1].plot(lims, lims, "k--", lw=1, label="perfect")
axes[1].set_xlabel("actual price (NT$/kg)")
axes[1].set_ylabel("predicted price (NT$/kg)")
axes[1].set_title("Actual vs predicted")
axes[1].legend()

plt.tight_layout()
plt.show()

# ══ 3.7 R² ═══════════════════════════════════════════════
# R² = 1 − SS_res / SS_tot
#   分母：什麼都不做、每天猜平均值會錯多少
#   分子：用模型預測會錯多少
#   1 − 比例 = 已經解決了多少比例的變異
#   R²=1 完美、R²=0 跟猜平均一樣、R²<0 比猜平均還差
# ⚠️ 這裡的數字有很大一部分是「價格會延續」這個常識，不是模型多聰明。
# 真正要看的是扣掉延續性之後還剩多少（3.9 節 E），以及時間切分的表現（3.11）。
r2 = 1 - np.sum((y_test - pred_test) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
print(f"測試資料 R² = {r2:.3f}（1 是完全貼合，0 是跟直接猜平均一樣）")
print(f"演進：最初版 0.826 → 加前一天價格 0.926 → 加量變化與交互項 {r2:.3f}")
