import pandas as pd
data = pd.read_csv(r'\data.csv')

pd.set_option('display.max_columns',None)
print(data.columns)
print(data.head(5))
print(data.describe())

features_mean = list(data.columns[2:12])
features_se = list(data.columns[12:22])
features_worst = list(data.columns[22:32])
data.drop("id",axis=1,inplace=True)  
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(data['diagnosis'],label='Count')
plt.show()

#corr()方法探究参数相关性，默认参数pearson用于定量数据正态分布时，spearman用于定量数据不满足正态时，kernell用于分类数据
corr = data[features_mean].corr()
sns.heatmap(corr,annot=True)
plt.figure(figsize=(14,14))
plt.show()

#根据热力图判定相关性高的因素组，组中保留1个因素即可。最终目的是让所选择的特征尽量独立、互不相关，增加模型泛化能力
features_remain = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean']
from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size=0.3)
train_X = train[features_remain]
train_y = train['diagnosis']
test_X = test[features_remain]
test_y = test['diagnosis']

#对数据进行归一标准化处理
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.transform(test_X)

#开始训练
import sklearn.svm as svm
from sklearn import metrics
model = svm.SVC()
model.fit(train_X,train_y)
prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
