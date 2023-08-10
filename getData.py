'''
Author: longshanxu 623119632@qq.com
Date: 2023-07-06 09:37:29
LastEditors: longshanxu 623119632@qq.com
LastEditTime: 2023-07-06 11:45:41
FilePath: \football_forecast_python\getData.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pymongo
import pandas as pd

# 连接MongoDB数据库
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["datacenter"]
collection = db["Money"]

# 查询所有数据并提取date字段
data = list(collection.find({}, {"date": 1,'_id':0}))

# 将数据转换为DataFrame对象
df = pd.DataFrame(data)

# 去重
df.drop_duplicates(inplace=True)

# 打印去重后的数据
# print(df)

# 遍历数据
for index, row in df.iterrows():
    print(row['date'])



