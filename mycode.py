import graphviz
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# 读取数据集
data = pd.read_csv("data1.csv")

# 将非数值型的特征进行独热编码
enc = OneHotEncoder()
X = enc.fit_transform(data.iloc[:, :-1]).toarray()
y = data.iloc[:, -1]

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建决策树分类器
model = DecisionTreeClassifier()

# 在训练集上训练模型
model.fit(X_train, y_train)

# 在测试集上评估模型的准确率
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)




# 预测新数据

new_data = pd.read_csv("data1_2.csv")

# 将非数值型的特征进行独热编码
new_X = enc.transform(new_data).toarray()

# 使用训练好的模型进行预测
new_y = model.predict(new_X)

# 打印预测结果
print("Prediction:", new_y)

dot_data = export_graphviz(model, out_file=None, 
                           feature_names=enc.get_feature_names_out(),  
                           class_names=['No', 'Yes'],  
                           filled=True, rounded=True,  
                           special_characters=True)

graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("decision_tree")
graph.view()