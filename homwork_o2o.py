# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:37:21 2018

@author: FNo0
"""

import pandas as pd
import xgboost as xgb
import warnings
import numpy as np
import time
import os

import joblib # 用于保存模型
from sklearn.model_selection import GridSearchCV,ParameterGrid # 用于网格搜索,与进度条配合使用

import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB  # 高斯朴素贝叶斯
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.neighbors import KNeighborsClassifier  # KNN,近邻算法
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.ensemble import AdaBoostClassifier  # AdaBoost
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.svm import SVC  # SVM
from sklearn.neural_network import MLPClassifier  # 多层感知机
from sklearn.ensemble import GradientBoostingClassifier  # GBDT,梯度提升决策树
from catboost import CatBoostClassifier # CatBoost,类别型特征
import lightgbm as lgb # lightGBM


warnings.filterwarnings('ignore')  # 不显示警告

# SVM等模型训练时间较长,用个进度条打印一下进度
class TqdmCallback:
    def __init__(self, total=None):
        self._total = total
        self._pbar = None

    def __call__(self, *args, **kwargs):
        if self._pbar is None:
            self._pbar = tqdm(total=self._total)
        elif self._pbar.n < self._total:
            self._pbar.update(1)

    def close(self):
        if self._pbar is not None:
            self._pbar.close()

"""数据预处理

    1.时间处理(方便计算时间差):
        将Date_received列中int或float类型的元素转换成datetime类型,新增一列date_received存储;
        将Date列中int类型的元素转换为datetime类型,新增一列date存储;

    2.折扣处理:
        判断折扣率是“满减”(如10:1)还是“折扣率”(0.9);
        将“满减”折扣转换为“折扣率”形式(如10:1转换为0.9);
        得到“满减”折扣的最低消费(如折扣10:1的最低消费为10);
    3.距离处理:
        将空距离填充为-1(区别于距离0,1,2,3,4,5,6,7,8,9,10);
        判断是否为空距离;

    Args:
        dataset: DataFrame类型的数据集off_train和off_test,包含属性'User_id','Merchant_id','Coupon_id','Discount_rate',
            'Distance','Date_received','Date'(off_test没有'Date'属性);

    Returns:
        预处理后的DataFrame类型的数据集.
    """
def prepare(dataset):
    
    # 源数据
    data = dataset.copy()
    # 折扣率处理
    data['is_manjian'] = data['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)  # Discount_rate是否为满减
    data['discount_rate'] = data['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))  # 满减全部转换为折扣率
    data['min_cost_of_manjian'] = data['Discount_rate'].map(
        lambda x: -1 if ':' not in str(x) else int(str(x).split(':')[0]))  # 满减的最低消费
    # 距离处理
    data['Distance'].fillna(-1, inplace=True)  # 空距离填充为-1
    data['null_distance'] = data['Distance'].map(lambda x: 1 if x == -1 else 0)
    # 时间处理
    data['date_received'] = pd.to_datetime(data['Date_received'], format='%Y%m%d')
    if 'Date' in data.columns.tolist():  # off_train
        data['date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    # 返回
    return data


"""打标

    领取优惠券后15天内使用的样本标签为1,否则为0;

    Args:
        dataset: DataFrame类型的数据集off_train,包含属性'User_id','Merchant_id','Coupon_id','Discount_rate',
            'Distance','Date_received','Date'

    Returns:
        打标后的DataFrame类型的数据集.
    """
def get_label(dataset):
   
    # 源数据
    data = dataset.copy()
    # 打标:领券后15天内消费为1,否则为0
    data['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0, data['date'],
                             data['date_received']))
    # 返回
    return data


def get_simple_feature(label_field):
    """简单的几个特征,作为初学示例

    Args:

    Returns:

    """
    # 源数据
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)  # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Date_received'] = data['Date_received'].map(int)  # 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float

	# 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['cnt'] = 1  # 添加了一个名为 'cnt' 的新列，并为所有行设置值为1
    # 返回的特征数据集
    feature = data.copy()

    # 用户领券数
    keys = ['User_id']  # 主键
    prefixs = 'simple_' + '_'.join(keys) + '_'  # 特征名前缀,由label_field和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()  # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')  # 将id列与特征列左连

    # 用户领取特定优惠券数
    keys = ['User_id', 'Coupon_id']  # 主键
    prefixs = 'simple_' + '_'.join(keys) + '_'  # 特征名前缀,由label_field和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()  # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')  # 将id列与特征列左连

    # 用户当天领券数
    keys = ['User_id', 'Date_received']  # 主键
    prefixs = 'simple_' + '_'.join(keys) + '_'  # 特征名前缀,由label_field和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()  # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')  # 将id列与特征列左连

    # 用户当天领取特定优惠券数
    keys = ['User_id', 'Coupon_id', 'Date_received']  # 主键
    prefixs = 'simple_' + '_'.join(keys) + '_'  # 特征名前缀,由label_field和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()  # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')  # 将id列与特征列左连

    # 用户是否在同一天重复领取了特定优惠券
    keys = ['User_id', 'Coupon_id', 'Date_received']  # 主键
    prefixs = 'simple_' + '_'.join(keys) + '_'  # 特征名前缀,由label_field和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt',
                           aggfunc=lambda x: 1 if len(x) > 1 else 0)  # 以keys为键,'cnt'为值,判断领取次数是否大于1
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'repeat_receive'}).reset_index()  # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')  # 将id列与特征列左连

    # 删除辅助提特征的'cnt'
    feature.drop(['cnt'], axis=1, inplace=True)

    # 返回
    return feature


def get_week_feature(label_field):
    """根据Date_received得到的一些日期特征

    根据date_received列得到领券日是周几,新增一列week存储,并将其one-hot离散为week_0,week_1,week_2,week_3,week_4,week_5,week_6;
    根据week列得到领券日是否为休息日,新增一列is_weekend存储;

    Args:

    Returns:

    """
    # 源数据
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)  # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Date_received'] = data['Date_received'].map(
        int)  # 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    # 返回的特征数据集
    feature = data.copy()
    feature['week'] = feature['date_received'].map(lambda x: x.weekday())  # 星期几
    feature['is_weekend'] = feature['week'].map(lambda x: 1 if x == 5 or x == 6 else 0)  # 判断领券日是否为休息日
    feature = pd.concat([feature, pd.get_dummies(feature['week'], prefix='week')], axis=1)  # one-hot离散星期几
    feature.index = range(len(feature))  # 重置index
    # 返回
    return feature


def get_dataset(history_field, middle_field, label_field):
    """构造数据集

    Args:

    Returns:

    """
    # 特征工程
    week_feat = get_week_feature(label_field)  # 日期特征
    simple_feat = get_simple_feature(label_field)  # 示例简单特征

    # 构造数据集
    share_characters = list(
        set(simple_feat.columns.tolist()) & set(week_feat.columns.tolist()))  # 共有属性,包括id和一些基础特征,为每个特征块的交集
    dataset = pd.concat([week_feat, simple_feat.drop(share_characters, axis=1)], axis=1)
    # 删除无用属性并将label置于最后一列
    if 'Date' in dataset.columns.tolist():  # 表示训练集和验证集
        dataset.drop(['Merchant_id', 'Discount_rate', 'Date', 'date_received', 'date'], axis=1, inplace=True)
        label = dataset['label'].tolist()
        dataset.drop(['label'], axis=1, inplace=True)
        dataset['label'] = label
    else:  # 表示测试集
        dataset.drop(['Merchant_id', 'Discount_rate', 'date_received'], axis=1, inplace=True)
    # 修正数据类型
    dataset['User_id'] = dataset['User_id'].map(int)
    dataset['Coupon_id'] = dataset['Coupon_id'].map(int)
    dataset['Date_received'] = dataset['Date_received'].map(int)
    dataset['Distance'] = dataset['Distance'].map(int)
    if 'label' in dataset.columns.tolist():
        dataset['label'] = dataset['label'].map(int)
    # 去重
    dataset.drop_duplicates(keep='first', inplace=True)
    dataset.index = range(len(dataset))
    # 返回
    return dataset

#分别定义['XGB','GNB', 'DT', 'KNN','RF', 'SVC', 'ada','LR', 'mlp','GBDT', 'CBC', 'LGBM']模型
def model_xgb(train, test):
    """xgb模型

    Args:

    Returns:

    """
    # xgb参数
    params = {'booster': 'gbtree', #booster [default=gbtree]，可选gbtree，gblinear或dart，这决定了使用哪种booster模型去训练
              'objective': 'binary:logistic',#目标函数，binary:logistic二分类的逻辑回归，返回预测的概率(不是类别),可选multi:softmax多分类问题，返回预测的类别(不是概率)，在这种情况下，你还需要多设一个参数：num_class(类别数目)
              'eval_metric': 'auc',#对于有效数据的度量方法，对于回归问题，默认值是rmse，对于分类问题，默认值是error。其它值有：rmse、mae、logloss、error、merror、mlogloss、auc
              'silent': 1,#当这个参数值为1时，静默模式开启，不会输出任何信息。一般这个参数就保持默认的0，因为这样能帮我们更好地理解模型
              'eta': 0.01,#和GBM中的learning rate参数类似。通过减少每一步的权重，可以提高模型的鲁棒性。典型值为0.01-0.2
              'max_depth': 5,#树的最大深度。缺省值为6，取值范围为：[1,∞]
              'min_child_weight': 1,#决定最小叶子节点样本权重和。和GBM的min_child_leaf参数类似，但不完全一样。XGBoost的这个参数是最小样本权重的和，而GBM参数是最小样本总数。这个参数用于避免过拟合，当它的值较大时，可以避免模型学习到局部的特殊样本。但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整
              'gamma': 0,#在节点分裂时，只有在分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。这个参数值越大，算法越保守。这个参数的值和损失函数息息相关，所以是需要调整的
              'lambda': 1,#权重的L2正则化项。这个参数是用来控制XGBoost的正则化部分的。虽然大部分数据科学家很少用到这个参数，但是这个参数在减少过拟合上还是可以挖掘出更多用处的
              'colsample_bylevel': 0.7,#决定每次节点分裂时，列采样的比例。XGBoost的默认值为1，也就是所有特征都使用。而GBM默认值为0.8。这个参数用于加速模型，特别是当你有很多列的时候
              'colsample_bytree': 0.7,#决定每棵树的列采样比例，和GBM里的max_features参数类似。用来控制每棵随机采样的列数的占比(每一列是一个特征)。缺省值为1，取值范围为：(0,1]，这个参数在不同的数据集上，起到了控制过拟合的作用。当然也会降低模型的学习速度。常用的值有：0.5-1
              'subsample': 0.9,#用来控制样本采样的比例，缺省值为1，取值范围为：(0,1]。和GBM的subsample参数一模一样。这个参数控制是否进行采样，让每棵树拥有不一样的数据实例。缺省值为1，表示不进行采样。值为0.5意味着每棵树都使用一半的训练数据。这个参数的值也要和learning_rate一同调整
              'scale_pos_weight': 1,#在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。平衡正负权重
              'nthread': 64 }# cpu线程数
    # print(params)
    print('Train a XGBoost model')
    time.sleep(3)
    # 数据集
    dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=train['label'])
    dtest = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))
    # 训练
    evals_result = {}
    watchlist = [(dtrain, 'train')]
    if os.path.exists('output_files/models/xgboost.pkl'):
        print('xgb loading...')
        model = joblib.load('output_files/models/xgboost.pkl')
        #调用模型的evals方法，对valid数据进行预测，得到预测值和真实值，然后调用evals方法，计算auc值
        model_eval = model.eval(dtrain)
        auc_values = model_eval.split()[1].split(':')[1]
        last_auc = float(auc_values)  # Get the last AUC value
    else:
        print('xgb training...')
        # model = xgb.train(params, dtrain, num_boost_round=5167, evals=watchlist, evals_result=evals_result, verbose_eval=False)
        model = xgb.train(params, dtrain, num_boost_round=5167, evals=watchlist, evals_result=evals_result, verbose_eval=True)
        joblib.dump(model, 'output_files/models/xgboost.pkl')
        auc_values = evals_result['train']['auc']
        last_auc = auc_values[-1]  # Get the last AUC value
    # 预测
    predict = model.predict(dtest)
    # 处理结果
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1) #合并用户id，优惠券id，收到优惠券日期，预测概率
    # 特征重要性
    feat_importance = pd.DataFrame(columns=['feature_name', 'importance'])
    feat_importance['feature_name'] = model.get_score().keys()
    feat_importance['importance'] = model.get_score().values()
    feat_importance.sort_values(['importance'], ascending=False, inplace=True)
    # 返回
    return result, feat_importance,last_auc

def model_gnb(train, validate):
    print('Train a GaussianNB model')
    # Prepare datasets
    X_train = train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)#去掉用户id，优惠券id，收到优惠券日期，标签，axis=1表示列
    y_train = train['label']
    X_validate = validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_validate = validate['label']
    # Train
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    # Predict
    pred = gnb.predict(X_validate)
    # Calculate accuracy
    accuracy = gnb.score(X_validate, y_validate)
    return accuracy

def model_dt(train, validate):
    print('Train a DecisionTree model')
    # Prepare datasets
    X_train = train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_train = train['label']
    X_validate = validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_validate = validate['label']
    # Train
    dtc = DecisionTreeClassifier(criterion ="entropy", max_depth = 4)
    dtc.fit(X_train, y_train)
    # Predict
    pred = dtc.predict(X_validate)
    # Calculate accuracy
    accuracy = dtc.score(X_validate, y_validate)
    return accuracy

def model_knn(train, validate):
    print('Train a KNeighbors model')
    # Prepare datasets
    X_train = train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_train = train['label']
    X_validate = validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_validate = validate['label']
    # Train
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    # Predict
    pred = knn.predict(X_validate)
    # Calculate accuracy
    accuracy = knn.score(X_validate, y_validate)
    return accuracy

def model_rf(train, validate):
    print('Train a RandomForest model')
    # Prepare datasets
    X_train = train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_train = train['label']
    X_validate = validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_validate = validate['label']
    # Train
    rf = RandomForestClassifier(n_estimators = 30)#n_estimators表示决策树的个数
    rf.fit(X_train, y_train)
    # Predict
    pred = rf.predict(X_validate)
    # Calculate accuracy
    accuracy = rf.score(X_validate, y_validate)
    return accuracy


    print('Train a AdaBoost model')
    # Prepare datasets
    X_train = train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_train = train['label']
    X_validate = validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_validate = validate['label']
    # Train
    ada = AdaBoostClassifier()
    ada.fit(X_train, y_train)
    # Predict
    pred = ada.predict(X_validate)
    # Calculate accuracy
    accuracy = ada.score(X_validate, y_validate)
    return accuracy

def model_svm(train, validate):
    print('Train a SVM model')
    # Prepare datasets
    train = train.sample(frac=0.1, random_state=123) # 为了加快训练速度，取10%的数据进行训练，random_state是随机数种子
    X_train = train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_train = train['label']
    X_validate = validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_validate = validate['label']
    # 网格搜索的参数
    # param_grid = { 
    # 'C': [0.1,],#0.1, 1, 10, 100, 1000
    # 'gamma': [1],#1, 0.1, 0.01, 0.001, 0.0001
    # 'kernel': ['linear']# 'linear', 'rbf', 'poly', 'sigmoid'
    # # 只保留一种参数，其他参数注释掉，我只是想打印train_score而已
    # }
    svm = SVC(C=0.1, gamma='scale', kernel='linear')

    if os.path.exists('output_files/models/svm.pkl'):
        svm = joblib.load('output_files/models/svm.pkl')
        print('Load SVM model...')
    else:
        print('Train SVM model...')
        svm = SVC(C=0.1, kernel='linear', gamma='scale', probability=True)
        svm.fit(X_train, y_train)
        joblib.dump(svm, 'output_files/models/svm.pkl')
    # Predict
    pred = svm.predict(X_validate)
    # Calculate accuracy
    accuracy = svm.score(X_validate, y_validate)
    return accuracy

def model_ada(train, validate):
    # Prepare datasets
    X_train = train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_train = train['label']
    X_validate = validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_validate = validate['label']
    if os.path.exists('output_files/models/ada.pkl'):
        ada = joblib.load('output_files/models/ada.pkl')
        print('Load AdaBoost model...')
    else:
        print('Train AdaBoost model...')
        ada = AdaBoostClassifier()
        ada.fit(X_train, y_train)
        joblib.dump(ada, 'output_files/models/ada.pkl')
    # Predict
    pred = ada.predict(X_validate)
    # Calculate accuracy
    accuracy = ada.score(X_validate, y_validate)
    return accuracy

def model_lr(train, validate):
    print('Train a LogisticRegression model')
    # Prepare datasets
    X_train = train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_train = train['label']
    X_validate = validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_validate = validate['label']
    if os.path.exists('output_files/models/lr.pkl'):
        lr = joblib.load('output_files/models/lr.pkl')
        print('Load LogisticRegression model...')
    else:
        print('Train LogisticRegression model...')
        lr = LogisticRegression(max_iter = 1200000)
        lr.fit(X_train, y_train)
        joblib.dump(lr, 'output_files/models/lr.pkl')
    # Predict
    pred = lr.predict(X_validate)
    # Calculate accuracy
    accuracy = lr.score(X_validate, y_validate)
    return accuracy

def model_mlp(train, validate):
    print('Train a MLP model')
    # Prepare datasets
    X_train = train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_train = train['label']
    X_validate = validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_validate = validate['label']
    if os.path.exists('output_files/models/mlp.pkl'):
        mlp = joblib.load('output_files/models/mlp.pkl')
        print('Load MLP model...')
    else:
        print('Train MLP model...')
        mlp = MLPClassifier(hidden_layer_sizes = (100, 100, 100), max_iter = 200)
        mlp.fit(X_train, y_train)
        joblib.dump(mlp, 'output_files/models/mlp.pkl')
    # Predict
    pred = mlp.predict(X_validate)
    # Calculate accuracy
    accuracy = mlp.score(X_validate, y_validate)
    return accuracy

def model_gbdt(train, validate):
    print('Train a GBDT model')
    # Prepare datasets
    X_train = train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_train = train['label']
    X_validate = validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_validate = validate['label']
    if os.path.exists('output_files/models/gbdt.pkl'):
        gbdt = joblib.load('output_files/models/gbdt.pkl')
        print('Load GBDT model...')
    else:
        print('Train GBDT model...')
        gbdt = GradientBoostingClassifier(logging_level='verbose')
        gbdt.fit(X_train, y_train)
        joblib.dump(gbdt, 'output_files/models/gbdt.pkl')
    # Predict
    pred = gbdt.predict(X_validate)
    # Calculate accuracy
    accuracy = gbdt.score(X_validate, y_validate)
    return accuracy

def model_cbc(train, validate):
    print('Train a cbc model')
    # Prepare datasets
    X_train = train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_train = train['label']
    X_validate = validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_validate = validate['label']
    if os.path.exists('output_files/models/cbc.pkl'):
        cbc = joblib.load('output_files/models/cbc.pkl')
        print('Load cbc model...')
    else:
        print('Train cbc model...')
        cbc = CatBoostClassifier(iterations = 1000, learning_rate = 0.1, depth = 6, loss_function = 'Logloss', eval_metric = 'AUC', logging_level = 'Verbose')  #logging_level='Verbose'表示输出训练过程中的信息
        cbc.fit(X_train, y_train)
        joblib.dump(cbc, 'output_files/models/cbc.pkl')
    # Predict
    pred = cbc.predict(X_validate)
    # Calculate accuracy
    accuracy = cbc.score(X_validate, y_validate)
    return accuracy

def model_lgb(train, validate):
    print('Train a LGBM model')
    # Prepare datasets
    X_train = train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_train = train['label']
    X_validate = validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)
    y_validate = validate['label']
    if os.path.exists('output_files/models/lgbm.pkl'):
        lgbm = joblib.load('output_files/models/lgbm.pkl')
        print('Load LGBM model...')
    else:
        print('Train LGBM model...')
        lgbm = lgb.LGBMClassifier(learning_rate = 0.01, max_bin = 150, num_leaves = 32, max_depth = 11, metric = 'auc', bagging_fraction = 0.8, feature_fraction = 0.8)
        lgbm.fit(X_train, y_train)
        joblib.dump(lgbm, 'output_files/models/lgbm.pkl')
    # Predict
    pred = lgbm.predict(X_validate)
    # Calculate accuracy
    accuracy = lgbm.score(X_validate, y_validate)
    return accuracy




if __name__ == '__main__':
    # 源数据
    
    off_train = pd.read_csv(r'dataset/ccf_offline_stage1_train.csv')
    off_test = pd.read_csv(r'dataset/ccf_offline_stage1_test_revised.csv')
    if not os.path.exists(r'output_files'):
        os.makedirs(r'output_files')
    if not os.path.exists(r'output_files/models'):
        os.makedirs(r'output_files/models')
    if not os.path.exists(r'output_files/figures'):
        os.makedirs(r'output_files/figures')
    # 预处理
    off_train = prepare(off_train)
    off_test = prepare(off_test)
    # 打标
    off_train = get_label(off_train)

    # 划分区间
    # 训练集历史区间、中间区间、标签区间
    train_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/3/2', periods=60))]  # [20160302,20160501)
    train_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/5/1', periods=15))]  # [20160501,20160516)
    train_label_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/5/16', periods=31))]  # [20160516,20160616)
    # 验证集历史区间、中间区间、标签区间
    validate_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/1/16', periods=60))]  # [20160116,20160316)
    validate_middle_field = off_train[
        off_train['date'].isin(pd.date_range('2016/3/16', periods=15))]  # [20160316,20160331)
    validate_label_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/3/31', periods=31))]  # [20160331,20160501)
    # 测试集历史区间、中间区间、标签区间
    test_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/4/17', periods=60))]  # [20160417,20160616)
    test_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/6/16', periods=15))]  # [20160616,20160701)
    test_label_field = off_test.copy()  # [20160701,20160801)

    # 构造训练集、验证集、测试集
    print('构造训练集')
    train = get_dataset(train_history_field, train_middle_field, train_label_field)
    print('构造验证集')
    validate = get_dataset(validate_history_field, validate_middle_field, validate_label_field)
    print('构造测试集')
    test = get_dataset(test_history_field, test_middle_field, test_label_field)

    #分别训练[xgb,gnb,dt,,knn,rf,svm,ada,lr,mlp,gbdt,cbc,lgb]

    #训练高斯朴素贝叶斯模型
    gnb_auc = model_gnb(train, validate)
    print("GaussianNB accuracy on validation set:", gnb_auc)

    #训练xgboost模型
    big_train = pd.concat([train, validate], axis=0)
    result, feat_importance,XGboost_auc = model_xgb(big_train, test)    
    result.to_csv(r'output_files/baseline.csv', index=False, header=None)
    feat_importance.to_csv(r'output_files/baseline_importance.csv', index=False)
    print('xgboost accuracy on validation set::', XGboost_auc)
    
    #训练决策树模型
    dt_auc = model_dt(train, validate)
    print("DecisionTree accuracy on validation set:", dt_auc)

    #训练KNN模型
    knn_auc = model_knn(train, validate)
    print("KNN accuracy on validation set:", knn_auc)

    #训练随机森林模型
    rf_auc = model_rf(train, validate)
    print("RandomForest accuracy on validation set:", rf_auc)

    #训练SVM模型
    svm_auc = model_svm(train, validate)
    print("SVM accuracy on validation set:", svm_auc)

    #训练AdaBoost模型
    ada_auc = model_ada(train, validate)
    print("AdaBoost accuracy on validation set:", ada_auc)

    #训练逻辑回归模型
    lr_auc = model_lr(train, validate)
    print("LogisticRegression accuracy on validation set:", lr_auc)

    #训练多层感知机模型
    mlp_auc = model_mlp(train, validate)
    print("MLP accuracy on validation set:", mlp_auc)

    #训练GBDT模型
    gbdt_auc = model_gbdt(train, validate)
    print("GBDT accuracy on validation set:", gbdt_auc)

    #训练CatBoost模型
    cbc_auc = model_cbc(train, validate)
    print("CatBoost accuracy on validation set:", cbc_auc)

    #训练LightGBM模型
    lgb_auc = model_lgb(train, validate)
    print("LightGBM accuracy on validation set:", lgb_auc)

    # 画图name=[xgb,gnb,dt,,knn,rf,svm,ada,lr,mlp,gbdt,cbc,lgb]
    name = ['gnb','xgb',  'dt', 'knn', 'rf', 'svm', 'ada', 'lr', 'mlp', 'gbdt', 'cbc', 'lgb']
    value=[gnb_auc,XGboost_auc,dt_auc,knn_auc,rf_auc,svm_auc,ada_auc,lr_auc,mlp_auc,gbdt_auc,cbc_auc,lgb_auc]
    #每一列不同颜色
    plt.figure(figsize=(25, 15))
    plt.bar(name, value, width=0.4)
    plt.xlabel('model')
    plt.ylabel('auc')
    plt.title('auc of different models')
    plt.savefig(r'output_files/figures/auc of different models.png')





