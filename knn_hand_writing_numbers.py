from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

#加载数据查看数据量、第一幅图信息及对应数字
digits = load_digits()
data = digits.data
print(data.shape)
print(digits.images[0])
print(digits.target[0])

#显示第一幅图
plt.gray()
plt.imshow(digits.images[0])

#拆分训练集和预测集
train_x,test_x,train_y,test_y = train_test_split(data,digits.target,test_size=0.25,random_state=33)

#数据标准化处理
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

#创建knn分类器
knn = KNeighborsClassifier()
knn.fit(train_ss_x,train_y)
predict_y = knn.predict(test_ss_x)
accuracy_score(test_y,predict_y)
