import pandas as pd
data = pd.read_csv(r'\UCI_Credit_Card.csv')
print(data.shape)

next_month = data['default.payment.next.month'].value_counts()
print(next_month)

#数据处理
data.drop(['ID'],inplace=True, axis=1)  #去除对分类无用的字段ID
target = data['default.payment.next.month'].values
columns = data.columns.tolist()
columns.remove('default.payment.next.month')
features = data[columns].values

#划分训练集和测试集
from sklearn.model_selection import train_test_split,GridSearchCV
#stratify指定某字段，意为区分训练集和测试集时根据该字段分层使之无偏
train_x,test_x,train_y,test_y = train_test_split(features,target,test_size=0.3,stratify=target,random_state=1)

#构造各类分类器
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

classifiers = [SVC(random_state = 1,kernel = 'rbf'),
              DecisionTreeClassifier(random_state = 1, criterion = 'gini'),
              RandomForestClassifier(random_state = 1, criterion = 'gini'),
              KNeighborsClassifier(metric='minkowski')] 

#分类器名称
classifier_names = ['svc','decisiontreeclassifier','randomforestclassifier','kneighborsclassifier']
#分类器参数
classifier_params = [{'svc__C':[1], 'svc__gamma':[0.01]},
                    {'decisiontreeclassifier__max_depth':[6,9,11]},
                    {'randomforestclassifier__n_estimators':[3,5,6]},
                    {'kneighborsclassifier__n_neighbors':[4,6,8]}]  

#对分类器进行GridSearchCV参数调优
from sklearn.metrics import accuracy_score
def GridSearchCV_work(pipeline,train_x,train_y,test_x,test_y,param_grid, score='accuracy'):
    response = {}
    gridsearch = GridSearchCV(estimator=pipeline,param_grid=param_grid,scoring=score)
    search = gridsearch.fit(train_x,train_y)
    print("GridSearch最优分类器及参数：", search.best_params_)
    print("GridSearch最优分数%0.4f"%search.best_score_)
    predict_y = gridsearch.predict(test_x)
    print("准确率",round(accuracy_score(test_y,predict_y),4))
    response['predict_y'] = predict_y
    response['accuracy_score'] = accuracy_score(test_y,predict_y)
    return response
    
    
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
for model, model_name, model_param_grid in zip(classifiers,classifier_names,classifier_params):
    pipeline = Pipeline([('scaler',StandardScaler()),(model_name,model)])
    result = GridSearchCV_work(pipeline,train_x,train_y,test_x,test_y,model_param_grid,score='accuracy')
