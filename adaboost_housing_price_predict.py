from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#使用sklearn自带加利福尼亚房价数据集，数据规范无需清洗规整
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()

train_x,train_y,test_x,test_y = train_test_split(data.data, data.target, test_size=0.25,random_state=33)

#使用Adaboost回归模型
regressor = AdaBoostRegressor()
regressor.fit(train_x,train_y)
predict_y = regressor.predict(test_x)
mse = mean_squared_error(test_y,predict_y)
print( predict_y)
print(round(mse,2))
