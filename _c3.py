# 1 資料：讀進來，整理成一張看得懂的表
# 這一格「不做任何修改」，只把資料看清楚：
#   有幾筆？每一欄是什麼型別？哪裡是空白？值域合不合理？有沒有整段缺月？
# 先看清楚才決定後面怎麼處理，這比一開始就急著補值或刪資料安全得多。

from pathlib import Path

import pandas as pd

# ── 1.1 找到檔案位置 ──────────────────────────────────────
# __vsc_ipynb_file__ 是 VS Code 提供的變數，指到這個 .ipynb 的完整路徑。
# 用它取 parent，資料夾就是 notebook 所在的那一層，
# 這樣不管從哪裡開啟 notebook，data/ 都找得到，不用寫死 C:\... 這種絕對路徑。
# 如果真的拿不到（例如貼到別的環境跑），就退回用「目前工作目錄」。
try:
    here = Path(__vsc_ipynb_file__).resolve().parent
except NameError:
    here = Path.cwd()

# ── 1.2 讀檔 ──────────────────────────────────────────────
# encoding="utf-8-sig" 是關鍵：這個 CSV 開頭帶有 BOM（\ufeff），
# 用一般的 "utf-8" 讀，第一欄名稱會變成 "\ufeff日期"，
# 之後 rename 或取欄位就會找不到而報 KeyError。
# utf-8-sig 會自動把 BOM 吃掉，欄名才是乾淨的「日期」。
raw = pd.read_csv(here / "data" / "taipei2_cabbage.csv", encoding="utf-8-sig")

# ── 1.3 改欄名 ────────────────────────────────────────────
# 原本是中文全名（「台北近30日均溫（℃）」這種又長又有全形括號），
# 打程式時很痛苦，也容易打錯一個字就整個 KeyError。
# 這裡一次換成短英文名，順便用 list(cols.values()) 固定欄位順序，
# 讓每一欄的位置永遠是 date, month, price, volume, temp30, rain7。
# 用 dict 對照而不是 df.columns = [...]，是因為萬一原檔欄位順序被改過，
# 用名字對照還是正確的；直接換位置則會把資料對錯欄。
cols = {
    "日期": "date",              # 交易日
    "月份": "month",             # 1~12，之後拿來做季節特徵
    "平均批發價（元/公斤）": "price",   # 要預測的目標
    "交易量（噸）": "volume",          # 當天賣出的總重量
    "台北近30日均溫（℃）": "temp30",   # 到當天為止 30 天平均氣溫
    "台北近7日雨量（毫米）": "rain7",   # 到當天為止 7 天累積雨量
}
df = raw.rename(columns=cols)[list(cols.values())]

# ── 1.4 基本體檢 ──────────────────────────────────────────
# 先確認「筆數對不對」，委託說有 796 天，這裡就該是 796。
# 再看日期範圍，確認涵蓋 2024~2026。
print("筆數 =", len(df), " 欄數 =", df.shape[1])
print("日期範圍", df["date"].min(), "到", df["date"].max())
# duplicated() 標記「這一列和前面某列重複」，
# 對 date 欄做就是檢查「同一天有沒有出現兩次」。
# 有的話代表資料被重複匯入，訓練時那幾天會被算兩次而失真。
# sum() 加總 True 的個數，0 就是沒問題。
print("日期有無重複 =", int(df["date"].duplicated().sum()))
print("\n欄位與型別")
# dtypes 確認「該是數字的欄位真的是數字」。
# 如果某欄被讀成 object（字串），通常是裡面混了 "-" 或空白之類，
# 那後面算平均、算 log 都會爆掉，要先用 pd.to_numeric 轉。
print(df.dtypes.to_string())

# ── 1.5 空白總表 ──────────────────────────────────────────
# isna() 產生同形狀的 True/False 表，sum() 沿著欄方向加總 → 每欄缺幾個。
blank = df.isna().sum()
# 用一張表同時看清四件事：
#   空白筆數：缺幾筆（絕對數量）
#   空白比例：缺幾 %（判斷嚴不嚴重，1% 和 40% 處理方式完全不同）
#   唯一值：distinct 個數，可以看 month 是不是只有 12 種、
#           也可以發現 price 有 397 種不同價格
# map 是對 Series 每個元素套用同一個函式，這裡把比例格式化成 "3.8%"。
report = pd.DataFrame({
    "空白筆數": blank,
    "空白比例": (blank / len(df)).map(lambda v: f"{v:.1%}"),
    "唯一值": df.nunique(),
})
display(report)

# any(axis=1) 是「這一列有任何一欄是空的嗎」；sum() 數出這種列有幾列。
print("有任何空白的列數 =", int(df.isna().any(axis=1).sum()))
# all(axis=1) 是「溫度和雨量兩個都空嗎」。
# 這數字很重要：如果等於「有任一個空白的列數」，
# 代表所有缺值都集中在同一批日子，處理時可以一次解決。
print("溫度、雨量同時空白的列數 =", int(df[["temp30", "rain7"]].isna().all(axis=1).sum()))

# ── 1.6 把缺值的列抓出來看 ────────────────────────────────
# 光看「缺 30 筆」還不夠，要看集中在哪裡。
# 如果集中在 2024 年初，可能是那段期間氣象站還沒接上；
# 如果散布在三年間，可能是隨機漏記。兩者處理方式不同。
display(df[df.isna().any(axis=1)].head(10))
# agg(["min","max"]) 一次拿到缺值日期的第一天和最後一天。
print(df.loc[df["temp30"].isna(), "date"].agg(["min", "max"]).to_string())

# ── 1.7 看資料本身 ────────────────────────────────────────
# 依日期排序後整張表印出來。reset_index(drop=True) 是重新編號 0~795，
# 不然排序後索引會是亂的，看起來很混淆。
display(df.sort_values("date").reset_index(drop=True))

# ── 1.8 統計摘要 ──────────────────────────────────────────
# describe() 給 count/mean/std/min/25%/50%/75%/max。
# 這裡要看的是「合理嗎」：
#   price 最小 4.8、最大 99.6 → 沒有負價、沒有 0 價
#   volume 最小 0.6 → 幾乎沒到貨的一天，之後會是預測誤差最大的那種
#   rain7 最大 361 → 颱風等級的雨量，是合理的極端值，不是打錯
# round(3) 只是讓畫面不要一堆小數。
display(df.describe().round(3))

# ── 1.9 每月天數 ──────────────────────────────────────────
# 如果某個月是 0 天，代表整段缺月（可能是檔案截斷）。
# 這裡每個月都有 50~77 天，說明資料是連續的。
# rename("天數").to_frame().T 是把 Series 轉成一列橫表，方便一眼看完 12 個月。
display(df.groupby("month").size().rename("天數").to_frame().T)
