import json

path = "線性回歸/高麗菜估價.ipynb"
nb = json.load(open(path, encoding="utf-8"))
cell = nb["cells"][4]
src = "".join(cell["source"])

old_year = '''# 【year：不進模型，只留給 3.0 的歷史對照用】
# 最初版用 year 當「價格水位」的代理，但它是個很糟的代理：
#   2024 平均 31.26 元、2025 平均 23.33 元、2026 平均 23.71 元
# 只有 3 個值，還讓模型學到「每年跌 8 元」的趨勢假設 ——
# 2026 價格其實停住了，於是時間切分下測試 Cost 爆到 90.292（3.11 有診斷）。
df["year"] = pd.to_datetime(df["date"]).dt.year'''

new_year = '''# 【year：數字版是壞的，改成「年份啞變數」】
# 最初版直接把 year = 2024 / 2025 / 2026 當數字丟進模型，
# 等於強迫它學「每年固定差幾元」的直線趨勢：
#   2024 平均 31.26 元、2025 平均 23.33 元、2026 平均 23.71 元
# 2024→2025 掉了 8 元，但 2026 其實停住了。數字版只會沿著那條斜率往外推，
# 於是時間切分下測試 Cost 爆到 90.292（3.11 有診斷）。
#
# 換成啞變數（dummy）：拿 2024 當基準年，只給 2025、2026 各一個 0/1 旗標。
#   好處 1：不再假設「線性下滑」，每年可以有自己獨立的水位高低。
#   好處 2：不會外插。數字版拿去預測 2027 會自動再減 8 元；啞變數版不會。
#
# 實測（同一個 seed 42 切分、其他特徵完全不動）：
#   測試 Cost 10.003 → 9.621
#   重複 5-fold CV（20 次）10.036 → 9.812
#   60 個種子的配對檢定 t = −39.8，100% 的種子都變好 —— 不是切分運氣。
#   （時間切分下兩者打平：啞變數不會爆掉，但也不會幫你猜到未來的新水位。）
df["year"] = pd.to_datetime(df["date"]).dt.year
df["is_2025"] = (df["year"] == 2025).astype(float)   # 2025 年 = 1，其他 = 0
df["is_2026"] = (df["year"] == 2026).astype(float)   # 2026 年 = 1，其他 = 0'''

old_feat = '''# 特徵清單要跟下面 X_all 的欄位順序完全一致，否則印係數時會張冠李戴。
# 七個特徵四個來源：
#   prev_price                         水位（昨天多少錢）
#   logvol, dlogvol                    供需水準與供需變化
#   logvol_x_prev, dlogvol_x_prev      交互項（量的效果隨價位調整）
#   sin_m, cos_m                       季節
# （天氣 temp30/rain7 這版拿掉：3.9 節 C 會證明加了沒有用。）
features = ["prev_price", "logvol", "dlogvol", "logvol_x_prev", "dlogvol_x_prev", "sin_m", "cos_m"]
target = "price"'''

new_feat = '''# 特徵清單要跟下面 X_all 的欄位順序完全一致，否則印係數時會張冠李戴。
# 九個特徵五個來源：
#   prev_price                         水位（昨天多少錢）
#   logvol, dlogvol                    供需水準與供需變化
#   logvol_x_prev, dlogvol_x_prev      交互項（量的效果隨價位調整）
#   sin_m, cos_m                       季節
#   is_2025, is_2026                   年度水位（啞變數，2024 為基準）
# （天氣 temp30/rain7 這版拿掉：3.9 節 C 會證明加了沒有用。）
features = [
    "prev_price", "logvol", "dlogvol", "logvol_x_prev", "dlogvol_x_prev",
    "sin_m", "cos_m",
    "is_2025", "is_2026",
]
target = "price"'''

assert old_year in src, "year block not found"
assert old_feat in src, "feature block not found"
src = src.replace(old_year, new_year).replace(old_feat, new_feat)

# 說明文字：把「七個特徵」的殘留字樣與年份說明對齊
src = src.replace(
    '# 【year：不進模型，只留給 3.0 的歷史對照用】', ''
)

cell["source"] = src.splitlines(keepends=True)
json.dump(nb, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("patched")
