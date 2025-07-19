import pandas as pd
import numpy as np

df = pd.read_csv('台灣各地區消費娛樂性產品.csv')
df = df.rename(columns={
    '1.食品及非酒精飲料' : '食品食品及非酒精飲料',
    '2.菸酒及檳榔' : '菸酒及檳榔',
    '3.衣著鞋襪及服飾用品' : '衣著鞋襪及服飾用品',
    '4.住宅服務、水電瓦斯及其他燃料' : '住宅服務、水電瓦斯及其他燃料',
    '5.家具設備及家務維護' : '家具設備及家務維護',
    '6.醫療保健' : '醫療保健',
    '7.交通' : '交通',
    '8.通訊' : '通訊',
    '9.休閒與文化' : '休閒與文化',
    '10.教育' : '教育',
    '11.餐廳及旅館' : '餐廳及旅館',
    '12.什項消費' : '什項消費',
    '(1)套裝旅遊(不含自助旅遊)': '套裝旅遊',
    '(2)娛樂消遣及文化服務': '娛樂消遣及文化服務',
    '(3)書報雜誌文具': '書報雜誌文具',
    '(4)教育消遣康樂器材及其附屬品': '教育消遣康樂器材及其附屬品'
})
# 分類維度表
# main_categories_raw = [
#     '食品及非酒精飲料',
#     '菸酒與及檳榔',
#     '衣著鞋襪及服飾用品',
#     '住宅服務、水電瓦斯及其他燃料',
#     '家具設備及家務維護',
#     '醫療保健',
#     '交通'
#     '通訊',
#     '休閒與文化',
#     '教育',
#     '餐廳及旅館',
#     '什項消費'
# ]
# category_ids = list(range(1, len(main_categories_raw) + 1))
# dim_data = {
#     'Category_ID': category_ids,
#     'Category_Name': main_categories_raw
# }
# dim_expense = pd.DataFrame(dim_data)
# dim_expense.to_csv("dim_expense.csv", index=False)

# 年份維度表
# year = []
# for i in range(2011,2024):
#     year.append(i)
# data = {
#     'year_id' : list(range(1, len(year) + 1)),
#     '年份' : year
# }
# dim_year = pd.DataFrame(data)
# dim_year.to_csv('dim_year.csv', index=False)

# 取消樞紐
# dftest = pd.melt(
#    df,
#    id_vars=['年份','地區'],
#    value_vars=['套裝旅遊', '娛樂消遣及文化服務', '書報雜誌文具','教育消遣康樂器材及其附屬品'],
#    var_name = '休閒與文化類別',  # 新的變數欄位名稱
#    value_name = '類別金額')


# 地區分類維度表
# Region = df['地區'].unique().tolist()
# data = {
#     'RegionID' : list(range(1, len(Region) + 1)),
#     'RegionName' : Region
# }
# north_regions = ['台北市', '新北市', '基隆市', '桃園市', '新竹市', '新竹縣', '宜蘭縣']
# central_regions = ['台中市', '苗栗縣', '彰化縣', '南投縣', '雲林縣']
# south_regions = ['台南市', '高雄市', '屏東縣', '嘉義市', '嘉義縣']
# east_regions = ['花蓮縣', '台東縣']
# island_regions = ['澎湖縣']
#
# conditions = [
#     df['地區'].isin(north_regions),
#     df['地區'].isin(central_regions),
#     df['地區'].isin(south_regions),
#     df['地區'].isin(east_regions),
#     df['地區'].isin(island_regions)
# ]
#
# # 3. 定義對應的選擇列表 (當條件滿足時要賦予的值)
# choices = ['北部', '中部', '南部', '東部', '離島']
# df['區域'] = np.select(conditions, choices, default='未分類')
# print("\n使用 np.select 新增 '區域' 欄位後的 df_combined (部分):\n", df)
#
# # 檢查是否有任何地區未被分類 (default 值)
# unclassified_regions = df[df['區域'] == '未分類']['地區'].unique()
# if len(unclassified_regions) > 0:
#     print(f"\n注意：以下地區未被分類到任何區域，請檢查您的地區列表: {unclassified_regions}")
# df.to_csv("台灣各地區消費娛樂性產品.csv", index=False, encoding='utf-8')

Region = df['地區'].unique().tolist()
data = {
    'RegionID' : list(range(1, len(Region) + 1)),
    'RegionName' : Region
}
dim_region = pd.DataFrame(data)
dim_region.to_csv('dim_region.csv', index=False)


