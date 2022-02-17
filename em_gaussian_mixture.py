import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

#数据加载避免中文乱码
data_ori = pd.read_csv(r'\heros.csv',encoding='gb18030')
features = [u'最大生命',u'生命成长',u'初始生命',u'最大法力', u'法力成长',u'初始法力',u'最高物攻',u'物攻成长',u'初始物攻',u'最大物防',
            u'物防成长',u'初始物防',u'最大每5秒回血', u'每5秒回血成长', u'初始每5秒回血', u'最大每5秒回蓝', u'每5秒回蓝成长',
            u'初始每5秒回蓝', u'最大攻速', u'攻击范围']
data = data_ori[features]

# 对英雄属性之间的关系进行可视化分析
# 设置 plt 正确显示中文
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

#用热力图呈现各属性间相关性
corr = data[features].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr,annot=True)
plt.show()

#相关性大的属性仅保留1个，以对属性值降维
features_remain = [u'最大生命', u'初始生命', u'最大法力', u'最高物攻', u'初始物攻', u'最大物防', u'初始物防', u'最大每5秒回血', 
                   u'最大每5秒回蓝', u'初始每5秒回蓝', u'最大攻速', u'攻击范围']
data = data_ori[features_remain]
data[u'最大攻速'] = data[u'最大攻速'].apply(lambda x:float(x.strip('%'))/100)
data[u'攻击范围'] = data[u'攻击范围'].map({'远程':1, '近战':0})

#规范化数据Z-score
ss = StandardScaler()
data = ss.fit_transform(data)

#构造GMM聚类
gmm=GaussianMixture(n_components=30,covariance_type='full')
gmm.fit(data)

#训练数据
prediction = gmm.predict(data)
print(prediction)

#将分组结果输出到CSV文件中
data_ori.insert(0,'test4',prediction)
data_ori.to_csv(r'd:\My Documents\Desktop\study\algorithm_data\EM_data-master\heros.csv',encoding='gb18030',sep=',')
