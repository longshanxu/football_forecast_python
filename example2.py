'''
Author: longshanxu 623119632@qq.com
Date: 2023-06-20 17:15:05
LastEditors: longshanxu 623119632@qq.com
LastEditTime: 2023-07-25 14:23:53
FilePath: \football_forecast_python\example2.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pymongo import MongoClient
import requests
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
import tabulate
import pydotplus
import tkinter as tk
import seaborn as sns


client = MongoClient('mongodb://localhost:27017/')
db = client['datacenter']

collection = db['AiData']

## 指定要忽略的字段
projection = {'_id': 0, '_created_at': 0 ,
              '_updated_at':0,'guestScore':0,
              'homeScore':0,'matchId':0,
              'matchTime':0,'date':0}

batch_size = 10000
data = []

for i in range(0, collection.count_documents({}), batch_size):
    batch = list(collection.find({}, projection).skip(i).limit(batch_size))
    data.extend(batch)

data = pd.DataFrame(data)

print("数据",len(data))

# ## 计算相关性矩阵
# corr_matrix = data.corr()
# print(corr_matrix)

# # 选择最相关的特征
# relevant_features = corr_matrix.index[abs(corr_matrix['result']) > 0.15]
# relevant_features = relevant_features.drop('result')
# print(relevant_features)
 
# 提取特征和标签
X = data.drop(['result'], axis=1)
# X = data[relevant_features]

y = data['result']

# # # 将数据集拆分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print("训练集大小：", len(X_train))
# print("测试集大小：", len(X_test))


# 打印包含NaN的列
nan_cols = data.columns[data.isna().any()].tolist()
print(nan_cols)

# nan_rows = data[data.isna().any(axis=1)]
# print(nan_rows)

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=18)
# rf = RandomForestClassifier(n_estimators=100, random_state=42)


# rf.fit(X_train, y_train)
rf.fit(X, y)


# # 创建梯度提升树模型
# gbt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)

# # 训练模型
# # gbt.fit(X, y)
# gbt.fit(X_train, y_train)

# # # 进行预测
# y_pred = gbt.predict(X_test)

# # 评估模型性能
# accuracy = accuracy_score(y_test, y_pred)
# print("模型准确率：", accuracy)

for i, tree in enumerate(rf.estimators_):
    print("Tree {}: Depth = {}".format(i, tree.tree_.max_depth))

# 进行预测
y_pred = rf.predict(X)


# 评估模型性能
accuracy = accuracy_score(y, y_pred)
print("模型准确率：", accuracy)

# # print(classification_report(y, y_pred))
# 计算混淆矩阵
cm = confusion_matrix(y, y_pred)
print("混淆矩阵：")
print(cm)

# 绘制箱线图
# sns.boxplot(data=X)
# plt.show()

# # 定义 K 折交叉验证
# kfold = KFold(n_splits=10, random_state=42, shuffle=True)

# # 进行交叉验证，并计算模型的准确率
# scores = cross_val_score(rf, X, y, cv=kfold)
# print("Cross-validation scores: ", scores)
# print("Average accuracy: ", scores.mean())


# # 获取随机森林中的第一个决策树
# tree = rf

# # 使用plot_tree函数绘制决策树模型
# import matplotlib.pyplot as plt
# plt.figure(figsize=(30,20), dpi=300)
# plot_tree(tree, feature_names=X.columns, filled=True, rounded=True)
# plt.show()
# plt.savefig('example2.png', dpi=300)

# importances = rf.feature_importances_
# indices = np.argsort(importances)[::-1]
# print("特征顺序：")
# for f in range(X.shape[1]):
#     print("%d. 特征名称：%s，重要性分数：%f" % (f + 1, X.columns[indices[f]], importances[indices[f]]))

# 特征重要性柱状图
importance = rf.feature_importances_
plt.barh(X.columns, importance)
plt.xticks(rotation=90)
plt.show()

# # 计算混淆矩阵
# cm = confusion_matrix(y , y_pred)

# # 打印混淆矩阵
# print("Confusion Matrix:")
# print(cm)


# # 计算准确率、精确率和召回率
# accuracy = (y_pred == y).mean()
# precision = cm[0, 0] / (cm[0, 0] + cm[1, 0] + cm[2, 0])
# recall = cm[0, 0] / (cm[0, 0] + cm[0, 1] + cm[0, 2])


# # 打印性能指标
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)


collection1 = db['ForeCastData']

## 指定要忽略的字段
projection = {'_id': 0, '_created_at': 0 ,'_updated_at':0,'guestScore':0 ,'homeScore':0,'prediction':0}

batch_size = 1000
dataNew = []

for i in range(0, collection1.count_documents({}), batch_size):
    batch = list(collection1.find({}, projection).skip(i).limit(batch_size))
    dataNew.extend(batch)

dfNew = pd.DataFrame(dataNew)

# 提取特征
X_test = dfNew.drop(['home1','guest2','matchId','matchTime','date'], axis=1)

# print(X_test)

# 进行预测
y_pred = rf.predict(X_test)


# print(X.columns)
# print(X_test.columns)   


# 将预测结果添加到新数据中
dfNew['prediction'] = y_pred

# # 创建GUI窗口
# root = tk.Tk()

# # 创建Text组件
# text = tk.Text(root)
# text.pack()

# # 在Text组件中打印DataFrame的内容
# text.insert(tk.END, dfNew[["home1","guest2","prediction"]].to_string())

# # 运行GUI窗口
# root.mainloop()

# 将DataFrame对象转换为JSON格式的字符串
json_data = dfNew[["matchId","prediction"]].to_json(orient="records")

# 将JSON格式的字符串转换为Python字典
data = {
    "data": json.loads(json_data)
}

# 将Python字典转换为JSON格式的字符串
json_data_with_data = json.dumps(data)

# 定义请求头
headers = {
    "Content-Type": "application/json",
    "X-Parse-Master-Key":"JsonMasterKey",
    "X-Parse-Application-Id":"JsonApp"
}

# 发送POST请求
response = requests.post("http://localhost/parse/functions/PythonRequest", data=json_data_with_data, headers=headers)

# 打印响应结果
print(response.text)