SEASON = ["sin_m", "cos_m"]
WEATHER = ["temp30", "rain7"]
PREV6 = ["prev_price", "logvol"] + WEATHER + SEASON   # 上一版的 6 特徵

ladder = [
    {"特徵組合": "笨基準：只猜平均價", "訓練 Cost": round(baseline_train, 3),
     "測試 Cost": round(baseline_test, 3), "特徵數": 0},
    row("只用 log(交易量)", ["logvol"]),
    row("logvol + 年份", ["logvol", "year"]),
    row("logvol + 天氣 + 年份（最初版）", ["logvol"] + WEATHER + ["year"]),
    row("logvol + 天氣 + 年份 + 季節", ["logvol"] + WEATHER + ["year"] + SEASON),
    row("+ 前一天價格", PREV6),
    row("+ 前一天交易量", PREV6 + ["prev_logvol"]),
    row("★ + 量變化與交互項（本版）", features),
    row("只用前一天價格", ["prev_price"]),
]
display(pd.DataFrame(ladder))
print("讀法：測試 Cost 越低越好。整張表是一條『一步步加對的欄位』的路線圖：")
print("  27.849 → 11.792（加前一天價格）→ 10.55（加前一天交易量）→ 10.00（加量變化與交互項）")
print("  每一段都不是調參數，而是想到『模型還缺哪一類資訊』。")
