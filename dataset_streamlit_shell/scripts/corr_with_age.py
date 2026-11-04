import pandas as pd

df = pd.read_csv('workspace/working.csv')

num_cols = ['是否生還', '船票等級', '同船兄弟姊妹與配偶數',
            '同船父母與子女數', '票價', '是否有船艙']
cat_cols = ['性別', '登船港口', '是否有船艙', '船票等級', '是否生還']

print("=== 數值欄位 vs 年齡：Pearson r ===")
res = []
for c in num_cols:
    sub = df[[c, '年齡']].dropna()
    r = sub[c].corr(sub['年齡'])
    res.append((c, round(r, 3), len(sub)))
for c, r, n in sorted(res, key=lambda x: -abs(x[1])):
    print(f"{c:>16}  r = {r:>7}   n = {n}")


def eta_squared(df, cat, num='年齡'):
    sub = df[[cat, num]].dropna()
    grand = sub[num].mean()
    ss_total = ((sub[num] - grand) ** 2).sum()
    ss_between = sum(
        len(g) * (g[num].mean() - grand) ** 2
        for _, g in sub.groupby(cat)
    )
    return ss_between / ss_total, len(sub), sub[cat].nunique()


print("\n=== 類別欄位 vs 年齡：相關比 eta^2 (0~1，越大越有關) ===")
for c in ['性別', '登船港口', '是否有船艙', '船票等級', '是否生還']:
    e, n, k = eta_squared(df, c)
    print(f"{c:>10}  eta^2 = {e:.4f}   n = {n}   類別數 = {k}")

print("\n=== 補充：艙等 x 性別 分組平均年齡 ===")
print(df.pivot_table(index='船票等級', columns='性別', values='年齡',
                     aggfunc=['mean', 'count']).round(1))
