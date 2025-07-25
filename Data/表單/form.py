import pandas as pd
df = pd.read_csv('潮玩市場調查.csv')
df = df.rename(columns={
    '請問您是否有聽說過潮流玩具?': '聽說過潮玩嗎?',
    '選擇潮牌玩具注重因素': '注重因素',
    '接觸潮玩主要來源': '接觸來源',
    '購買潮玩產品預算': '購買預算',
    '購買潮玩主要原因': '購買原因',
    '是否有意願 購買 / 重複購買': '有意願購買/重複購買',
    '是否有購買過潮流玩具': '曾購買潮玩',
    '是否購買分類': '購買分類' # 這個欄位看起來已經是處理過的分類
})

# 將 '聽說潮玩', '有意願購買/重複購買', '曾購買潮玩' 轉換為布林值或數值
df['聽說過潮玩嗎?'] = df['聽說過潮玩嗎?'].map({'是': True, '否': False})
df['有意願購買/重複購買'] = df['有意願購買/重複購買'].map({'是': True, '否': False})
df['曾購買潮玩'] = df['曾購買潮玩'].map({'是': True, '否': False})

# 定義預算區間到數值的映射
budget_mapping = {
    '500元以下': 250,
    '500–1000元': 750,
    '1000–3000元': 2000,
    '超過3000元': 3500 # 取一個代表值
}
df['購買預算_數值'] = df['購買預算'].map(budget_mapping)

df.insert(0, '受訪者編號', range(1, len(df) + 1))

df_reason = df[['受訪者編號', '購買原因']].dropna()
df_reason['購買原因'] = df_reason['購買原因'].str.split(',')
df_reason = df_reason.explode('購買原因')
df_reason.to_csv("潮玩_購買原因_關聯表.csv", index=False,encoding='utf-8-sig')

brand_cols = ['POPMART', 'Disney', "Pok'emon"]
df_brand = df[['受訪者編號'] + brand_cols]
df_brand = df_brand.melt(id_vars='受訪者編號', var_name='品牌', value_name='填答')
df_brand = df_brand[df_brand['填答'].notna()].drop(columns='填答')
df_brand.to_csv("潮玩_品牌偏好_關聯表.csv", index=False)

records = []

for brand in brand_cols:
    temp = df[['受訪者編號', brand]].dropna()
    temp['商品'] = temp[brand].str.split(',')
    temp = temp.explode('商品')
    temp['商品'] = temp['商品'].str.strip()
    temp['品牌'] = brand
    records.append(temp[['受訪者編號', '品牌', '商品']])
    # 合併所有品牌的商品記錄
df_brand_product = pd.concat(records, ignore_index=True)

# 儲存成 CSV
df_brand_product.to_csv("品牌_購買商品_關聯表.csv", index=False,encoding='utf-8-sig')

# 儲存清洗後的資料到新的 CSV 檔案
df.to_csv('潮玩市場調查_清洗後.csv', index=False, encoding='utf-8-sig')



