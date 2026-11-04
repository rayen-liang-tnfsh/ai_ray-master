import json
import pandas as pd

path = "dataset_streamlit_shell/workspace/working.csv"
df = pd.read_csv(path)
print("before:", list(df.columns))

rename_map = {
    "PassengerId": "乘客編號",
    "Survived": "是否生還",
    "Pclass": "船票等級",
    "Name": "姓名",
    "Sex": "性別",
    "Age": "年齡",
    "SibSp": "同船兄弟姊妹與配偶數",
    "Parch": "同船父母與子女數",
    "Ticket": "船票號碼",
    "Fare": "票價",
    "Cabin": "船艙編號",
    "Embarked": "登船港口",
}

df = df.rename(columns=rename_map)
df.to_csv(path, index=False)

print("after:", list(df.columns))
print("shape:", df.shape)

log = {
    "created_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "actor": "agent",
    "action": "rename_columns_to_zh_tw",
    "columns": list(rename_map.values()),
    "rows": int(len(df)),
    "note": "將 12 個英文欄位名稱改為繁體中文（乘客編號、是否生還、船票等級、姓名、性別、年齡、同船兄弟姊妹與配偶數、同船父母與子女數、船票號碼、票價、船艙編號、登船港口）。",
}
with open("dataset_streamlit_shell/workspace/cleaning_log.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(log, ensure_ascii=False) + "\n")
print("log appended")
