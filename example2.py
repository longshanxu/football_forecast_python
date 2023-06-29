'''
Author: longshanxu 623119632@qq.com
Date: 2023-06-20 17:15:05
LastEditors: longshanxu 623119632@qq.com
LastEditTime: 2023-06-29 20:13:37
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
projection = {'_id': 0, '_created_at': 0 ,'_updated_at':0,'guestScore':0,'homeScore':0,}

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

# 将数据集分割为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# for i, tree in enumerate(rf.estimators_):
#     print("Tree {}: Depth = {}".format(i, tree.tree_.max_depth))

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

# 定义随机森林分类器
# rf = RandomForestClassifier(n_estimators=100, random_state=42)

# # 进行交叉验证
# scores = cross_val_score(rf, X, y, cv=5)

# # 输出交叉验证结果
# print("Cross-validation scores: ", scores)
# print("Average score: ", np.mean(scores))



# 定义参数分布
# param_dist = {
#     'n_estimators': np.arange(50, 500, 50),
#     'max_depth': np.arange(3, 10)
# }

# # 定义随机森林分类器
# rf = RandomForestClassifier()

# # 定义随机搜索对象
# random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, cv=5)

# # 进行随机搜索
# random_search.fit(X, y)

# # 输出最优的参数组合和准确率
# print("Best parameters: ", random_search.best_params_)
# print("Best score: ", random_search.best_score_)

# # 定义参数网格
# param_grid = {
#     'n_estimators': [50, 100, 200,300],
#     'max_depth': [3, 5, 7,9]
# }

# # 定义随机森林分类器
# rf = RandomForestClassifier()

# # 定义网格搜索对象
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)

# # 进行网格搜索
# grid_search.fit(X, y)

# # 输出最优的参数组合和准确率
# print("Best parameters: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)

# 热力图
# sns.heatmap(rf.feature_importances_.reshape(1, -1), cmap="YlGnBu", annot=True, cbar=False, xticklabels=X.columns)
# plt.show()

# 特征重要性柱状图
# importance = rf.feature_importances_
# plt.bar(X.columns, importance)
# plt.xticks(rotation=90)
# plt.show()

# 生成网格点
# xx, yy = np.meshgrid(np.arange(X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1, 0.1),
#                      np.arange(X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1, 0.1))

# # 预测网格点的标签
# Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# # 绘制决策边界
# plt.contourf(xx, yy, Z, alpha=0.4)
# plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, alpha=0.8)
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()

# 计算混淆矩阵
cm = confusion_matrix(y, y_pred)

# 打印混淆矩阵
print("Confusion Matrix:")
print(cm)


# # 绘制混淆矩阵可视化
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.colorbar()
# tick_marks = np.arange(len([0,1,2]))
# plt.xticks(tick_marks, [0,1,2], rotation=45)
# plt.yticks(tick_marks, [0,1,2])
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')

# # 显示图像
# plt.show()

# 计算分类报告
# target_names = ['class 0', 'class 1', 'class 2']
# report = classification_report(y, y_pred, target_names=target_names, output_dict=True)

# # 将分类报告转换为表格
# table_data = []
# for k, v in report.items():
#     if k in target_names:
#         row = [k, v['precision'], v['recall'], v['f1-score'], v['support']]
#         table_data.append(row)
# table_data.append(['', '', '', '', ''])
# table_data.append(['accuracy', '', '', report['accuracy'], ''])

# # 绘制分类报告可视化
# fig, ax = plt.subplots()
# ax.axis('off')
# ax.axis('tight')
# ax.table(cellText=table_data, colLabels=['class', 'precision', 'recall', 'f1-score', 'support'], loc='center')

# # 显示图像
# plt.show()

# # 找出错误样本的索引
# idx = np.where(y != y_pred)[0]

# 绘制散点图
# plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y)
# plt.scatter(X.iloc[idx, 0], X.iloc[idx, 1], c='r', marker='x')
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()

# # 获取错误样本的特征
# X_error = X.iloc[idx]

# # 计算特征之间的相关系数
# corr = X_error.corr()

# # 绘制热力图
# sns.heatmap(corr, cmap='coolwarm')

# # 显示图像
# plt.show()


# # 绘制散点图
# plt.scatter(X, y)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# # 绘制带有回归线的散点图
# sns.lmplot(x="x", y="y", data=data)
# plt.show()

# 计算准确率、精确率和召回率
accuracy = (y_pred == y).mean()
precision = cm[0, 0] / (cm[0, 0] + cm[1, 0] + cm[2, 0])
recall = cm[0, 0] / (cm[0, 0] + cm[0, 1] + cm[0, 2])




# 打印性能指标
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)


# accuracy = np.trace(cm) / float(np.sum(cm))
# print('Accuracy:', accuracy)

# # 计算分类报告
# target_names = ['class 0', 'class 1', 'class 2']
# print("Classification Report:")
# print(classification_report(y, y_pred, target_names=target_names))

# # 打印混淆矩阵
# print("Confusion Matrix:")
# print(cm)

# # 打印分类报告
# report = classification_report(y, y_pred)
# print("Classification Report:")
# print(report)

# 生成分类性能报告
# report = classification_report(y_test, y_pred)

# # 打印分类性能报告
# print("Classification Report:")
# print(report)

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
X_test = dfNew.drop(['home1','guest2'], axis=1)

# 进行预测
y_pred = rf.predict(X_test)

# for i in range(len(y_pred)):
#     query = {'_id': i}
#     update = {'$set': {'prediction': y_pred[i]}}
#     collection.update_one(query, update)


# # 将预测结果添加到新数据中
dfNew['prediction'] = y_pred


# # 将dfNew中的数据写回到MongoDB中
# data = dfNew.to_dict(orient='records')
# collection1.delete_many({})
# collection1.insert_many(data)

# 创建GUI窗口
root = tk.Tk()

# 创建Text组件
text = tk.Text(root)
text.pack()

# 在Text组件中打印DataFrame的内容
text.insert(tk.END, dfNew[["home1","guest2","prediction"]].to_string())

# 运行GUI窗口
root.mainloop()
