'''
Author: longshanxu 623119632@qq.com
Date: 2023-06-20 15:01:22
LastEditors: longshanxu 623119632@qq.com
LastEditTime: 2023-06-20 15:37:05
FilePath: \football_forecast_python\example1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
from pymongo import MongoClient
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text

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


relevant_features = list(data.columns)
relevant_features.remove('result')


X = data[relevant_features]
y = data['result']


# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
clf = DecisionTreeClassifier(criterion='entropy', max_features=0.5)
clf.fit(X_train, y_train)

# 测试模型
score = clf.score(X_test, y_test)
print("Accuracy:", score)

# 输出决策树
tree_rules = export_text(clf, feature_names=relevant_features)
print(tree_rules)