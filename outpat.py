'''
Author: longshanxu 623119632@qq.com
Date: 2023-06-12 19:05:51
LastEditors: longshanxu 623119632@qq.com
LastEditTime: 2023-06-16 14:08:55
FilePath: \vue_vuetify_parseserver_cypress\src\python\outpat.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import graphviz
from matplotlib import pyplot as plt
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text, plot_tree
from sklearn.metrics import accuracy_score
import seaborn as sns
from tabulate import tabulate

# 读取Mongo的数据
client = MongoClient('mongodb://localhost:27017/')
db = client['datacenter']

collection = db['AiData']

## 指定要忽略的字段
projection = {'_id': 0, '_created_at': 0 ,'_updated_at':0,'guestScore':0,'homeScore':0,}

batch_size = 1000
data = []

for i in range(0, collection.count_documents({}), batch_size):
    batch = list(collection.find({}, projection).skip(i).limit(batch_size))
    data.extend(batch)

data = pd.DataFrame(data)

data.dropna(inplace=True)


# print(data.describe())

# print(data.isnull().sum())

# ## 计算相关性矩阵
# corr_matrix = data.corr()

# # 选择最相关的特征
# relevant_features = corr_matrix.index[abs(corr_matrix['result']) > 0.2]
# print(relevant_features)

relevant_features = list(data.columns)
relevant_features.remove('result')


X = data[relevant_features]
y = data['result']

# # 创建决策树模型
# model = DecisionTreeClassifier()

# # 训练模型
# model.fit(X, y)

# # 输出特征权重
# for feature, importance in zip(relevant_features, model.feature_importances_):
#     print(feature, importance)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建决策树分类器
model = RandomForestClassifier()

# 在训练集上训练模型
model.fit(X_train, y_train)

# 在测试集上评估模型的准确率
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# print(export_text(model, feature_names=X_train.columns.tolist()))

print('预测结果:', y_pred)


# 使用plot_tree函数绘制决策树模型
# import matplotlib.pyplot as plt
# plt.figure(figsize=(20,10))
# plot_tree(model, feature_names=relevant_features, filled=True, rounded=True)
# plt.show()

# 获取随机森林中的第一个决策树
tree = model.estimators_[0]

# 使用plot_tree函数绘制决策树模型
import matplotlib.pyplot as plt
plt.figure(figsize=(30,20), dpi=600)
plot_tree(tree, feature_names=relevant_features, filled=True, rounded=True)
plt.show()
# plt.savefig('example.png', dpi=600)

# collection = db['ForeCastData']

# ## 指定要忽略的字段
# projection = {'_id': 0, '_created_at': 0 ,'_updated_at':0,'guestScore':0 ,'homeScore':0, 'result':0}

# batch_size = 1000
# dataNew = []

# for i in range(0, collection.count_documents({}), batch_size):
#     batch = list(collection.find({}, projection).skip(i).limit(batch_size))
#     dataNew.extend(batch)

# dfNew = pd.DataFrame(dataNew)

# dfNew.dropna(inplace=True)

# y_predNew = model.predict(dfNew[relevant_features])

# print('预测结果:', y_predNew)

# # 将预测结果添加到新数据中
# dfNew['predicted_homeScore'] = y_predNew

# # 将数据转换为表格形式
# table = tabulate(dfNew[["home1", "guest2", "predicted_homeScore"]], headers='keys', tablefmt='psql')

# # 打印表格
# print(table)