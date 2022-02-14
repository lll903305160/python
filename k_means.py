from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np

data = pd.read_csv(r'data.csv',encoding='gbk')
data.info()

#输入数据
train_x = data[["2019年国际排名","2018世界杯","2015亚洲杯"]]
df = pd.DataFrame(train_x)
kmeans = KMeans(n_clusters=3)

#数据规范化
min_max_scaler = preprocessing.MinMaxScaler()
train_x = min_max_scaler.fit_transform(train_x)

#引入kmeans算法
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
predict_y

#合并预测数据到原数据中
result = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
print(result)
