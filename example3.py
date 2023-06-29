'''
Author: longshanxu 623119632@qq.com
Date: 2023-06-20 17:15:05
LastEditors: longshanxu 623119632@qq.com
LastEditTime: 2023-06-29 18:12:21
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

collection = db['AiQiuData']

## 指定要忽略的字段
projection = {'_id': 0, '_created_at': 0 ,'_updated_at':0,}

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

# 进行预测
y_pred = rf.predict(X)

# # 计算特征重要性得分
# importances = rf.feature_importances_

# # # 绘制特征重要性可视化
# plt.bar(range(X.shape[1]), importances)
# plt.xticks(range(X.shape[1]), X, rotation=0)
# plt.xlabel('Features')
# plt.ylabel('Importance Score')
# plt.title('Feature Importance')
# plt.show()


# 打印随机森林的决策深度
# for i, tree in enumerate(rf.estimators_):
#     print('Depth of decision tree', i+1, ':', tree.tree_.max_depth)

# 计算混淆矩阵
cm = confusion_matrix(y, y_pred)

# 打印混淆矩阵
print("Confusion Matrix:")
print(cm)


# 计算准确率、精确率和召回率
accuracy = (y_pred == y).mean()


# 打印性能指标
print("Accuracy:", accuracy)



