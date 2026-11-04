# 2 前處理
# 這一格做五件事，順序不能亂：
#   (0) 依日期排序 → (1) 造特徵 → (2) 處理空白 → (3) 切訓練/測試 → (4) 縮放尺度
# 為什麼順序重要：
#   第 (0) 步是「前一天的價格/交易量」這些 lag 特徵的前提，shift() 只認「上一列」，
#   資料沒照日期排好，拿到的「上一列」就不是真的前一天；
#   空白必須在「切之前」處理掉，因為你不想讓測試資料影響任何決定；
#   縮放用的平均/標準差必須在「切之後」只從訓練資料算，否則就是資料洩漏。
# 做完之後，訓練只會看到 X_train、y_train；X_test 留到最後才見光。

import numpy as np

# ══ 2.0 先依日期排序 ═════════════════════════════════════
# shift(1) 的定義是「往上一列搬」，它完全不知道日期，只認表格裡的上下位置。
# 如果資料沒照日期排好，shift(1) 就會安靜地拿到別天的值 ——
# 這種錯不會報錯，只會讓模型看起來超準，是最危險的 bug。
# 所以寧可多寫這一行，把「日期由舊到新」變成明確的前提。
df = df.sort_values("date").reset_index(drop=True)

# ══ 2.1 造特徵 ═══════════════════════════════════════════
#
# 【為什麼交易量要取 log】
# 交易量和價格的關係是「倍數」關係不是「加法」關係：
# 50→100 噸 和 25→50 噸 對價格的影響應該一樣，取 log 後都變成 +log2。
# 取完 log 相關係數從 -0.808 升到 -0.860，關係更直，線性模型更好用。
df["logvol"] = np.log(df["volume"])

# 【前一天的價格 —— 第一次改動，也是最重要的一欄】
# 委託寫「一天收市後，你已經知道當天的交易量和天氣」。
# 收市後的那個時間點，昨天的收盤價當然也已知，所以它不算作弊、也不是事後解釋。
# 相鄰兩天價格相關高達 0.94；平均每天只動 3.63 元，但價格標準差 17.57 元。
# 意思是：真正大的是「整體水位」，不是每天亂跳。
# 給模型昨天的價格 = 直接告訴它「現在大概在哪個價位帶」。
df["prev_price"] = df["price"].shift(1)

# 【前一天的交易量 & 今天的交易量變化 —— 第二次改動】
# 同一天的量和價是一起決定的，所以「今天量多少」大多解釋「今天價位在哪」；
# 真正決定今天價格往上還是往下的是「今天比昨天多/少賣多少」。
df["prev_logvol"] = df["logvol"].shift(1)
df["dlogvol"] = df["logvol"] - df["prev_logvol"]          # 正 = 今天量變多

# 【交互項：讓「量的效果」隨價位調整】
# 同樣多賣 10 噸，在 10 元的低價位和 60 元的高價位，衝擊不可能一樣大。
# 交互項是線性模型裡表達「效果會變動」最簡單的方式。
df["logvol_x_prev"] = df["logvol"] * df["prev_price"]
df["dlogvol_x_prev"] = df["dlogvol"] * df["prev_price"]

# 【季節用 sin/cos】月份是循環的（12 月和 1 月是鄰居），
# 直接丟 month=1..12 會讓模型以為 1 月和 12 月差很遠。
# 放到圓上用 (sin, cos) 座標表示就保留了循環性，兩個數字就代表 12 個月。
df["sin_m"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_m"] = np.cos(2 * np.pi * df["month"] / 12)

# 【year：不進模型，只留給 3.0 的歷史對照用】
# 最初版用 year 當「價格水位」的代理，但它是個很糟的代理：
#   2024 平均 31.26 元、2025 平均 23.33 元、2026 平均 23.71 元
# 只有 3 個值，還讓模型學到「每年跌 8 元」的趨勢假設 ——
# 2026 價格其實停住了，於是時間切分下測試 Cost 爆到 90.292（3.11 有診斷）。
df["year"] = pd.to_datetime(df["date"]).dt.year

# 特徵清單要跟下面 X_all 的欄位順序完全一致，否則印係數時會張冠李戴。
# 七個特徵四個來源：
#   prev_price                         水位（昨天多少錢）
#   logvol, dlogvol                    供需水準與供需變化
#   logvol_x_prev, dlogvol_x_prev      交互項（量的效果隨價位調整）
#   sin_m, cos_m                       季節
# （天氣 temp30/rain7 這版拿掉：3.9 節 C 會證明加了沒有用。）
features = ["prev_price", "logvol", "dlogvol", "logvol_x_prev", "dlogvol_x_prev", "sin_m", "cos_m"]
target = "price"

# ══ 2.2 處理空白 ═════════════════════════════════════════
#
# 【決定：整列剔除，不補值】
# 溫度缺 30 天（3.8%）、雨量缺 11 天（1.4%），比例很小。
# 補值（填平均、內插）填進去的都是「猜的」，但模型會當成真的觀測。
# 天氣有連日性（連續下雨、連續寒流），用平均填會把那些日子
# 硬生生變成「不冷不熱不雨」的正常日，扭曲分布。丟掉的資訊遠小於失真的風險。
#
# 剔除時看的欄位：
#   volume、temp30、rain7（原始觀測）
#   + prev_price、prev_logvol（排序後第一列沒有「前一天」，必然是空的）
# 不列 price 是因為它本來就沒缺；其餘欄位都是從母欄算出來的，母欄沒問題就沒問題。
#
# 剛好第一列（2024-01-02）的 temp30 本身就是空的，
# 所以 lag 特徵只多砍了「本來就要砍的那一列」，列數仍是 766。
before = len(df)
clean = df.dropna(subset=["volume", "temp30", "rain7", "prev_price", "prev_logvol"]).reset_index(drop=True)
# reset_index(drop=True)：剔除後重新編號 0~765，
# 不然索引會是 0,1,2,...,6,8,9（7 被砍掉），後面用位置索引時會錯亂。
print(f"原始 {before} 天，保留 {len(clean)} 天，剔除 {before - len(clean)} 天")
print(f"（前一天的價格/交易量各缺 {int(df['prev_price'].isna().sum())} 天，只有排序後的第一列）")

# 轉成 numpy：矩陣運算比 DataFrame 快，也不會有索引對齊的意外。
X_all = clean[features].to_numpy(dtype=float)
y_all = clean[target].to_numpy(dtype=float)

# ══ 2.3 切訓練集與測試集 ═════════════════════════════════
#
# 【為什麼要留測試集】
# 拿全部資料訓練再拿同一批算 Cost，只證明「模型背得出答案」。
# 留 20% 從頭到尾不給模型看，那個 Cost 才是「遇到沒見過的日子有多準」。
#
# 【為什麼固定種子 42】
# 不固定的話每次重跑切分都不一樣，Cost 會飄（11.8 / 12.1 / 11.6），
# 就無法判斷「改進後從 28.3 降到 11.8」是改進有效還是切到好日子。
# 固定種子 = 每次切同一批日子 = 版本之間可公平比較。
#
# 委託要求「隨機留 20% 的日子當測試」，所以這裡用隨機切。
# 但隨機切對時間相依的資料會偏樂觀（測試日期的隔天常在訓練集），
# 所以 3.11 會另外做「時間切分」對照，那個數字才是真實預測力。
rng = np.random.default_rng(42)
order = rng.permutation(len(clean))
n_test = int(round(0.2 * len(clean)))       # 20% → 153 天
test_idx = order[:n_test]
train_idx = order[n_test:]                 # 其餘 613 天
X_train_raw, y_train = X_all[train_idx], y_all[train_idx]
X_test_raw, y_test = X_all[test_idx], y_all[test_idx]

# ══ 2.4 z-score 縮放 ═════════════════════════════════════
#
# 【為什麼要縮放】
# 特徵尺度差很多：prev_price 5~100、logvol 約 3.5、交互項到 77、sin/cos 是 −1~1。
# 不縮放的話，梯度會被大尺度欄位放大，同一個 alpha 對某些欄位太大（暴衝）、
# 對某些欄位太小（走不動）。縮放後每欄都是「平均 0、標準差 1」，alpha 才能共用。
#
# 【公式】x_scaled = (x − mu) / sigma
#
# ⚠️ mu、sigma 只能從「訓練資料」算。
# 用全部資料算的話，測試集的分布就偷偷流進訓練流程（資料洩漏），
# 測試 Cost 會偏樂觀。真實世界裡未來的資料根本還不存在，不可能拿來算平均。
mu = X_train_raw.mean(axis=0)
sigma = X_train_raw.std(axis=0)
X_train = (X_train_raw - mu) / sigma
X_test = (X_test_raw - mu) / sigma        # 測試資料用「訓練的 mu、sigma」縮放

# ══ 2.5 檢查 ═════════════════════════════════════════════
print(f"訓練 {len(X_train)} 天，測試 {len(X_test)} 天")
print("形狀：X_train", X_train.shape, " y_train", y_train.shape,
      " X_test", X_test.shape, " y_test", y_test.shape)

# mu、sigma 印出來：之後預測新的一天必須用「這一組」數字縮放，不能重算（3.10 會用到）。
display(pd.DataFrame({"特徵": features, "訓練平均 mu": mu, "訓練標準差 sigma": sigma}).round(4))

# 驗算：縮放後訓練資料每欄應該是 平均≈0、標準差≈1。
# 這行在「抓錯」：如果 mu/sigma 算錯欄或誤用測試資料，數字就會對不上。
print("縮放後訓練資料的平均（應接近 0）", np.round(X_train.mean(axis=0), 6))
print("縮放後訓練資料的標準差（應接近 1）", np.round(X_train.std(axis=0), 6))
