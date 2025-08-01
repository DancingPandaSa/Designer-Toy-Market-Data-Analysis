import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import os

# ==============================
# 工具函式
# ==============================
def clean_numeric_columns(df):
    """自動清理數值欄位：去除逗號、轉 float、補缺值"""
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False).astype(float)
            except ValueError:
                continue
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
    return df

def train_rf_with_rfecv(X, y, cv, use_rfecv=True):
    """訓練 RF + RFECV，返回模型與特徵"""
    base_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    if use_rfecv:
        selector = RFECV(estimator=base_model, step=1, cv=cv, scoring='r2', n_jobs=-1)
        selector.fit(X, y)
        selected_features = X.columns[selector.support_].tolist()
    else:
        base_model.fit(X, y)
        importances = pd.Series(base_model.feature_importances_, index=X.columns)
        selected_features = importances.nlargest(len(X.columns) // 2).index.tolist()
    final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(X[selected_features], y)
    return final_model, selected_features

def evaluate_model(model, X, y):
    """模型評估 (MSE, R2)"""
    y_pred = model.predict(X)
    return mean_squared_error(y, y_pred), r2_score(y, y_pred)

# ==============================
# 主流程函式
# ==============================
def two_stage_forecast(input_csv, use_rfecv=True):
    df = pd.read_csv(input_csv)
    df = clean_numeric_columns(df)

    # --- 自動判斷類別欄位 ---
    categorical_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in ['年份']]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    # --- 自動選取數值欄位 ---
    numeric_cols = [c for c in df_encoded.columns if c not in ['年份'] + list(df_encoded.columns[df_encoded.columns.str.contains('_')])]

    # --- 階段一特徵（家庭結構類） ---
    stage1_target_features = [c for c in numeric_cols if any(k in c for k in ['家庭戶數', '平均每戶'])]
    stage1_input_features = ['年份'] + [c for c in df_encoded.columns if any(x in c for x in categorical_cols)]
    X_stage1, Y_stage1 = df_encoded[stage1_input_features], df_encoded[stage1_target_features]

    print("\n=== 階段一：預測家庭結構特徵 ===")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    stage1_models, stage1_selected = {}, {}
    for feature in stage1_target_features:
        model, selected = train_rf_with_rfecv(X_stage1, Y_stage1[feature], cv, use_rfecv)
        stage1_models[feature] = model
        stage1_selected[feature] = selected
        print(f"[階段一] {feature} -> 選特徵 {len(selected)}")

    # --- 準備 2024 輸入 ---
    latest_year = df_encoded['年份'].max()
    predict_2024_input_stage1 = df_encoded[df_encoded['年份'] == latest_year][[c for c in stage1_input_features if c != '年份']].copy()
    predict_2024_input_stage1['年份'] = 2024
    predicted_2024_features = pd.DataFrame({
        f: stage1_models[f].predict(predict_2024_input_stage1[stage1_selected[f]])
        for f in stage1_target_features
    }, index=predict_2024_input_stage1.index)

    # --- 階段二：消費支出 ---
    stage2_targets = [c for c in numeric_cols if c not in stage1_target_features]
    X_stage2 = df_encoded[stage1_input_features + stage1_target_features]
    Y_stage2 = df_encoded[stage2_targets]

    print("\n=== 階段二：預測消費娛樂性產品支出 ===")
    stage2_models, stage2_selected, stage2_eval = {}, {}, {}
    for target in stage2_targets:
        model, selected = train_rf_with_rfecv(X_stage2, Y_stage2[target], cv, use_rfecv)
        stage2_models[target] = model
        stage2_selected[target] = selected
        mse, r2 = evaluate_model(model, X_stage2[selected], Y_stage2[target])
        stage2_eval[target] = {'MSE': mse, 'R2': r2}
        print(f"[階段二] {target} -> R2: {r2:.3f}")

    # --- 準備 2024 輸入 ---
    predict_2024_input_stage2 = predict_2024_input_stage1.copy()
    for col in stage1_target_features:
        predict_2024_input_stage2[col] = predicted_2024_features[col]
    predicted_2024_consumption = pd.DataFrame({
        t: stage2_models[t].predict(predict_2024_input_stage2[stage2_selected[t]])
        for t in stage2_targets
    }, index=predict_2024_input_stage2.index)

    # --- 合併結果 ---
    final_2024_predictions = pd.concat([
        pd.DataFrame({'年份': [2024] * len(predicted_2024_features)}, index=predicted_2024_features.index),
        predicted_2024_features,
        predicted_2024_consumption
    ], axis=1)

    # --- 自動命名輸出檔 ---
    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    output_filename = f"2024年_{base_name}_兩階段預測結果.csv"
    final_2024_predictions.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n=== 完成！已輸出：{output_filename} ===")
    return final_2024_predictions

# ==============================
# 執行：換檔名即可
# ==============================
# two_stage_forecast('台灣各地區消費娛樂性產品.csv', use_rfecv=False)
# two_stage_forecast('台灣性別消費娛樂性產品.csv', use_rfecv=True)
two_stage_forecast('C:\\Designer-Toy-Market-Data-Analysis\\Data\\台灣資料整理\\台灣各年齡消費娛樂性產品.csv', use_rfecv=True)
