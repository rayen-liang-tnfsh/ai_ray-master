import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DFKai-SB", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

here = Path(r"c:\Users\user\Documents\GitHub\ai_ray-master\邏輯回歸")
games = pd.read_csv(here / "data" / "mlb_games.csv")
games.columns = games.columns.str.strip()
df = games.copy()
df["失誤差（次）"] = df["對手失誤（次）"] - df["失誤（次）"]
FEATURES = ["安打差（支）", "失誤差（次）", "主場（1為主場）"]
X_raw = df[FEATURES].to_numpy(float)
y = df["贏球（1為贏）"].to_numpy(int)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)
b = clf.intercept_[0]
w = clf.coef_[0]
print("截距 b =", round(b, 6))
print(f"{'欄位':<14}{'係數 w':>9}{'勝算比':>10}")
for n, wi in zip(FEATURES, w):
    print(f"{n:<14}{wi:>9.4f}{np.exp(wi):>10.4f}")
print("訓練準確率 =", round(accuracy_score(y, clf.predict(X)), 4))


def std_of(i, raw):
    return (raw - scaler.mean_[i]) / scaler.scale_[i]


hd = np.linspace(-12, 12, 241)
z_hit = std_of(0, hd)
z_err = std_of(1, 0)
z_home = std_of(2, 1)
z_away = std_of(2, 0)


def prob(zh):
    return 1 / (1 + np.exp(-(b + w[0] * z_hit + w[1] * z_err + w[2] * zh)))


p_home = prob(z_home)
p_away = prob(z_away)

home_rows = df[df["主場（1為主場）"] == 1]
cut = pd.cut(home_rows["安打差（支）"], bins=np.arange(-12, 13, 2))
emp = home_rows.groupby(cut, observed=True)["贏球（1為贏）"].agg(["mean", "size"])
emp_x = [iv.mid for iv in emp.index]
emp_n = emp["size"].to_numpy()
emp_p = emp["mean"].to_numpy()
print("分箱（安打差中點, 實測勝率, 樣本數）:")
for a, bp, cn in zip(emp_x, emp_p, emp_n):
    print(f"  {a:>6}  {bp:.3f}  n={cn}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(hd, p_home, lw=2.2, label="模型：主場球隊")
ax.plot(hd, p_away, lw=2.2, ls="--", label="模型：客隊")
ax.scatter(emp_x, emp_p, s=np.clip(emp_n * 1.5, 20, 220), color="black", alpha=0.55, zorder=3, label="實際勝率")
ax.axhline(0.5, color="gray", lw=0.8, ls=":")
ax.axvline(0, color="gray", lw=0.8, ls=":")
ax.set_xlabel("安打差（支，己方減對手）")
ax.set_ylabel("贏球機率")
ax.set_title("安打差與贏球機率（失誤差固定在平均）")
ax.set_ylim(0, 1)
ax.grid(alpha=0.25)
ax.legend(loc="lower right")
plt.tight_layout()
out = here / "images" / "win_prob.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=120)
print("saved:", out)

# 注意：一場比賽的兩列是同一場，客隊的安打差 = 主隊安打差的相反數。
# 所以要驗「兩邊機率相加 = 1」，必須把客隊那條曲線左右翻轉來對：
print()
print("同一場比賽的兩隊（主隊安打差 +h，客隊就是 -h）：")
for v in (-10, -5, -3, 0, 3, 5, 10):
    k = np.abs(hd - v).argmin()
    km = np.abs(hd + v).argmin()
    print(f"  安打差 {v:>3}：主場 {p_home[k]:.4f} + 客隊 {p_away[km]:.4f} = {p_home[k] + p_away[km]:.4f}")

print()
print("若只看同一支球隊、同樣安打差，主場 vs 客場的差別：")
for v in (-3, 0, 3, 5):
    k = np.abs(hd - v).argmin()
    print(f"  安打差 {v:>3}：主場 {p_home[k]:.4f}，客場 {p_away[k]:.4f}，差 {p_home[k] - p_away[k]:+.4f}")
