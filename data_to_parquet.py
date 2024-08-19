import pandas as pd

# 讀取Excel文件
READ_ROUTE = "repo/train_data_5.xlsx"
WRITE_ROUTE = "data/train_data_8.parquet"
# 前綴
prompt = "要求：請閱讀文章後根據文章主旨以後面提供的標準將文章進行分類：如果文章的內容完全不涉及地層下陷或地面下陷問題，請歸類為「非描述地層下陷問題之文章」。如果文章涉及施工導致的事件，例如路面塌陷、工地事故，或是關於施工過程中的損害處理、安全管理措施、修復策略或事後的應對和改進措施，請歸類為「工地災害：損害處理、安全管理、復修策略、後續回應」。如果文章討論的是地層下陷問題，但這些問題不是由施工安全問題導致的，而是由自然事件（如地震、大雨沖刷、地基塌陷等）造成，或是起因不明的下陷事件，或是對這類事件(非施工相關)的後續應對和改進措施，請歸類為「非施工安全導致之地層下陷問題、後續及其應對措施」。文章如下： "

df = pd.read_excel(READ_ROUTE)

# 對"input"欄位進行前綴添加操作
df["input"] = df["input"].apply(lambda x: prompt + x)

# 重命名欄位
df = df.rename(columns={"input": "instruction"})

# 選取所需欄位並添加一個空的"input"欄位
df["input"] = ""
df = df[["instruction", "input", "output"]]

# 定義格式化函數
def format_to_parquet(row):
    return f"<|start_header_id|>user<|end_header_id|>{row['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{row['output']}<|eot_id|>"

# 將每行應用轉換函數
df['text'] = df.apply(format_to_parquet, axis=1)

# 刪除不需要的欄位
df.drop(['instruction', 'output', 'input'], axis=1, inplace=True)

# 保存為Parquet文件
df.to_parquet(WRITE_ROUTE, engine='pyarrow')

print("成功轉換並保存為Parquet。")
