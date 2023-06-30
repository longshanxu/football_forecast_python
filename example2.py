'''
Author: longshanxu 623119632@qq.com
Date: 2023-06-20 17:15:05
LastEditors: longshanxu 623119632@qq.com
LastEditTime: 2023-06-30 14:10:23
FilePath: \football_forecast_python\example2.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
import tabulate
import tkinter as tk
import seaborn as sns


client = MongoClient('mongodb://localhost:27017/')
db = client['datacenter']

collection = db['AiData']

## 指定要忽略的字段
projection = {'_id': 0, '_created_at': 0 ,'_updated_at':0,'guestScore':0,'homeScore':0,'matchId':0,'matchTime':0,'date':0}

batch_size = 10000
data = []

for i in range(0, collection.count_documents({}), batch_size):
    batch = list(collection.find({}, projection).skip(i).limit(batch_size))
    data.extend(batch)

data = pd.DataFrame(data)

print("数据",len(data))



# 提取特征和标签
X = data.drop('result', axis=1)
y = data['result']

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

for i, tree in enumerate(rf.estimators_):
    print("Tree {}: Depth = {}".format(i, tree.tree_.max_depth))

# 进行预测
y_pred = rf.predict(X)


# 特征重要性柱状图
importance = rf.feature_importances_
plt.bar(X.columns, importance)
plt.xticks(rotation=90)
plt.show()

# 计算混淆矩阵
cm = confusion_matrix(y, y_pred)

# 打印混淆矩阵
print("Confusion Matrix:")
print(cm)


# 计算准确率、精确率和召回率
accuracy = (y_pred == y).mean()
precision = cm[0, 0] / (cm[0, 0] + cm[1, 0] + cm[2, 0])
recall = cm[0, 0] / (cm[0, 0] + cm[0, 1] + cm[0, 2])


# 打印性能指标
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)


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

# 创建GUI窗口
root = tk.Tk()

# 创建Text组件
text = tk.Text(root)
text.pack()

# 在Text组件中打印DataFrame的内容
text.insert(tk.END, dfNew[["home1","guest2","prediction"]].to_string())

# 运行GUI窗口
root.mainloop()
