import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
# 讀取數據集
# df = pd.read_csv('台灣各地區消費娛樂性產品.csv')
# df = pd.read_csv('台灣性別消費娛樂性產品.csv')
df = pd.read_csv('台灣各年齡消費娛樂性產品.csv')
# --- 數據清洗與預處理 ---
# 定義所有數值型欄位，包括特徵和目標，以便去除逗號並轉換為浮點數
all_numeric_cols = [
    '家庭戶數', '平均每戶人數', '平均每戶成年人數', '平均每戶就業人數', '平均每戶所得收入者人數',
    '所得收入總計', '總消費支出', '食品食品及非酒精飲料', '菸酒及檳榔',
    '衣著鞋襪及服飾用品', '住宅服務、水電瓦斯及其他燃料', '家具設備及家務維護', '醫療保健',
    '交通', '通訊', '休閒與文化', '教育', '餐廳及旅館', '什項消費',
    '套裝旅遊', '娛樂消遣及文化服務', '書報雜誌文具', '教育消遣康樂器材及其附屬品'
]

for col in all_numeric_cols:
    if col in df.columns:
        # 使用 errors='coerce' 將無法轉換的值設為 NaN，然後再處理 NaN
        df[col] = df[col].astype(str).str.replace(',', '', regex=False).astype(float)

# 處理缺失值 (這裡簡單地用中位數填充，也可以根據實際情況選擇其他策略)
for col in all_numeric_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# 執行 One-Hot 編碼，不丟棄第一個類別，以確保所有類別在預測時都能對應
df_encoded = pd.get_dummies(df, columns=['年齡'], drop_first=False)

# --- 階段一：預測 2024 年的「特徵數據」 ---
# 階段一的目標：需要預測的非消費/收入類數值特徵
stage1_target_features = [
    '家庭戶數', '平均每戶人數', '平均每戶成年人數',
    '平均每戶就業人數', '平均每戶所得收入者人數'
]

# 階段一的輸入特徵 年份 + (地區、性別、年齡)的 One-Hot 編碼
# 確保這些特徵在 df_encoded 中存在
stage1_input_features_base = ['年份']
# stage1_input_features_categorical = [col for col in df_encoded.columns if '地區_' in col or '區域_' in col]
# stage1_input_features_categorical = [col for col in df_encoded.columns if '性別_' in col]
stage1_input_features_categorical = [col for col in df_encoded.columns if '年齡_' in col]
stage1_input_features = list(set(stage1_input_features_base + stage1_input_features_categorical))

X_stage1_train = df_encoded[stage1_input_features]
Y_stage1_train = df_encoded[stage1_target_features]
# 為每個階段一目標訓練模型
stage1_models = {}
stage1_selected_features = {}
print("\n--- 階段一：開始預測 2024 年的特徵數據 ---")
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
for i, feature_col in enumerate(stage1_target_features):
    print(f"\n--- 處理階段一目標: {feature_col} ({i + 1}/{len(stage1_target_features)}) ---")
    y_target_stage1 = Y_stage1_train[feature_col]

    estimator_stage1 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    selector_stage1 = RFECV(estimator=estimator_stage1, step=1, cv=cv_strategy,
                            scoring='r2', n_jobs=-1)
    selector_stage1.fit(X_stage1_train, y_target_stage1)
    selected_feats = X_stage1_train.columns[selector_stage1.support_].tolist()
    stage1_selected_features[feature_col] = selected_feats
    print(f"為 '{feature_col}' 選定的特徵數量: {len(selected_feats)}")

    final_model_stage1 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    final_model_stage1.fit(X_stage1_train[selected_feats], y_target_stage1)
    stage1_models[feature_col] = final_model_stage1
# 準備 2024 年的階段一預測輸入
# 我們需要 2024 年的地區和區域信息。這裡我們假設 2024 年的地區和區域分佈與最新年份（2023）相同。
latest_year_data_for_template = df_encoded[df_encoded['年份'] == df_encoded['年份'].max()].copy()
predict_2024_input_stage1 = latest_year_data_for_template[stage1_input_features_categorical].copy()
predict_2024_input_stage1['年份'] = 2024
# 確保欄位順序與訓練時一致
predict_2024_input_stage1 = predict_2024_input_stage1[X_stage1_train.columns]


# 預測 2024 年的特徵數據
predicted_2024_features = pd.DataFrame(index=predict_2024_input_stage1.index)
for feature_col in stage1_target_features:
    model_stage1 = stage1_models[feature_col]
    selected_feats = stage1_selected_features[feature_col]
    predicted_2024_features[feature_col] = model_stage1.predict(predict_2024_input_stage1[selected_feats])

print("\n--- 階段一：2024 年特徵數據預測完成 ---")


# --- 階段二：使用預測的 2024 年特徵數據，預測「消費娛樂性產品支出」 ---

# 階段二的目標：消費/收入相關欄位
stage2_target_columns = [
    '所得收入總計', '總消費支出', '食品食品及非酒精飲料', '菸酒及檳榔',
    '衣著鞋襪及服飾用品', '住宅服務、水電瓦斯及其他燃料', '家具設備及家務維護', '醫療保健',
    '交通', '通訊', '休閒與文化', '教育', '餐廳及旅館', '什項消費',
    '套裝旅遊', '娛樂消遣及文化服務', '書報雜誌文具', '教育消遣康樂器材及其附屬品'
]
# 階段二的輸入特徵：年份 + 預測的階段一特徵 + 地區/區域的 One-Hot 編碼
# 構建完整的階段二訓練特徵集 X_stage2_train
# 包含所有原始的基礎特徵 (包括年份) 和 One-Hot 編碼後的地區/區域特徵
X_stage2_train_base = df_encoded[stage1_input_features + stage1_target_features] # 這裡包含了年份和所有階段一的特徵
Y_stage2_train = df_encoded[stage2_target_columns]

# 儲存每個階段二目標的模型、選定的特徵和評估指標
stage2_models = {}
stage2_selected_features = {}
evaluation_metrics_stage2 = {}

print("\n--- 階段二：開始預測 2024 年的消費娛樂性產品支出 ---")

for i, target_col in enumerate(stage2_target_columns):
    print(f"\n--- 處理階段二目標: {target_col} ({i+1}/{len(stage2_target_columns)}) ---")

    y_target_stage2 = Y_stage2_train[target_col]

    estimator_stage2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    selector_stage2 = RFECV(estimator=estimator_stage2, step=1, cv=cv_strategy,
                            scoring='r2', n_jobs=-1)
    selector_stage2.fit(X_stage2_train_base, y_target_stage2)

    selected_feats_stage2 = X_stage2_train_base.columns[selector_stage2.support_].tolist()
    stage2_selected_features[target_col] = selected_feats_stage2
    print(f"為 '{target_col}' 選定的特徵數量: {len(selected_feats_stage2)}")

    final_model_stage2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    final_model_stage2.fit(X_stage2_train_base[selected_feats_stage2], y_target_stage2)
    stage2_models[target_col] = final_model_stage2

    # 評估模型在訓練集上的性能
    y_pred_train_stage2 = final_model_stage2.predict(X_stage2_train_base[selected_feats_stage2])
    mse_stage2 = mean_squared_error(y_target_stage2, y_pred_train_stage2)
    r2_stage2 = r2_score(y_target_stage2, y_pred_train_stage2)
    evaluation_metrics_stage2[target_col] = {'MSE': mse_stage2, 'R2': r2_stage2}
    print(f"訓練集 MSE: {mse_stage2:.2f}, R2: {r2_stage2:.2f}")
# 準備 2024 年的階段二預測輸入
# 結合 2024 年份、地區/區域 One-Hot 編碼、以及預測的 2024 年特徵數據
predict_2024_input_stage2 = predict_2024_input_stage1.copy() # 包含年份和地區/區域 One-Hot
# 將預測的階段一特徵合併進來
for col in stage1_target_features:
    predict_2024_input_stage2[col] = predicted_2024_features[col]

# 確保 predict_2024_input_stage2 的欄位順序和名稱與 X_stage2_train_base 完全一致
predict_2024_input_stage2 = predict_2024_input_stage2[X_stage2_train_base.columns]


# 預測 2024 年的消費娛樂性產品支出
predicted_2024_consumption = pd.DataFrame(index=predict_2024_input_stage2.index)
for target_col in stage2_target_columns:
    model_stage2 = stage2_models[target_col]
    selected_feats_stage2 = stage2_selected_features[target_col]
    predicted_2024_consumption[target_col] = model_stage2.predict(predict_2024_input_stage2[selected_feats_stage2])

print("\n--- 階段二：2024 年消費娛樂性產品支出預測完成 ---")
# 將預測結果與 2024 年的基礎資訊結合
# 為了最終輸出 CSV 時包含地區和區域，我們從原始 df 中獲取這些信息
# 注意：這裡的索引需要與 predict_2024_input_stage1 的索引保持一致
final_2024_predictions_df = pd.DataFrame({
    '年份': 2024,
    # '地區': df.loc[latest_year_data_for_template.index, '地區'].values,
    # '區域': df.loc[latest_year_data_for_template.index, '區域'].values
    # '性別': df.loc[latest_year_data_for_template.index, '性別'].values
    '年齡': df.loc[latest_year_data_for_template.index, '年齡'].values
}, index=predicted_2024_features.index) # 使用 predicted_2024_features 的索引來對齊

# 合併預測的階段一特徵
final_2024_predictions_df = pd.concat([final_2024_predictions_df, predicted_2024_features], axis=1)
# 合併預測的階段二目標
final_2024_predictions_df = pd.concat([final_2024_predictions_df, predicted_2024_consumption], axis=1)


# --- 將最終預測結果匯出為 CSV 檔案 ---
# output_filename = '2024年台灣各地區消費娛樂性產品_兩階段預測結果.csv'
# output_filename = '2024年台灣性別消費娛樂性產品_兩階段預測結果.csv'
output_filename = '2024年台灣年齡消費娛樂性產品_兩階段預測結果.csv'
final_2024_predictions_df.to_csv(output_filename, index=False, encoding='utf-8-sig') # 使用 utf-8-sig 確保中文顯示正常
print(f"\n2024 年兩階段預測結果已匯出至 '{output_filename}'")

print("\n--- 兩階段預測完成 ---")
# print("此預測是基於以下假設：")
# print("1. 2024 年的地區和區域分佈與數據集中最新年份（2023 年）相同。")
# print("2. 2024 年的非消費/收入特徵（如家庭戶數、平均每戶人數等）是透過模型預測而來，而非簡單複製 2023 年的數據。")
# print("這種兩階段方法通常能提供更精確的未來預測，尤其是當特徵本身也隨時間變化時。")




