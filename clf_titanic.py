import pandas as pd
train_data = pd.read_csv(r'\train.csv')
test_data = pd.read_csv(r'\test.csv')

# 数据探索
print(train_data.info())
print('-'*30)
print(train_data.describe())
print('-'*30)
print(train_data.describe(include=['O']))
print('-'*30)
print(train_data.head())
print('-'*30)
print(train_data.tail())

# 数据清洗
train_data['Age'].fillna(train_data['Age'].mean(),inplace = True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace = True)
train_data['Fare'].fillna(train_data['Fare'].mean(),inplace = True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace = True)
train_data['Embarked'].fillna('S',inplace=True)
test_data['Embarked'].fillna('S',inplace=True)

# 特征选择
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

#字符串向量化
from sklearn.feature_extraction import DictVectorizer
dvec = DictVectorizer(sparse=False)
train_features = dvec.fit_transform(train_features.to_dict(orient='record'))

#构造决策树
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = 'entropy')
clf.fit(train_features,train_labels)

#对测试集的字符串向量化处理
test_features = dvec.fit_transform(test_features.to_dict(orient='record'))

#进行决策树预测,因测试集中无survived字段无法测试，所以用训练集测试
pred_labels = clf.predict(test_features)
acc_decision_tree = round(clf.score(train_features,train_labels),4)
acc_decision_tree

#为了避免以上这种自测导致准确率很高的情况，对训练集采取k折交叉验证
import numpy as np
from sklearn.model_selection import cross_val_score
acc_k = np.mean(cross_val_score(clf,train_features,train_labels,cv=10))
acc_k
