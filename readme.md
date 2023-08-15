# 本代码实现了12种常用方法，没有调参，精度打印如下：
```
Train a GaussianNB model
GaussianNB accuracy on validation set: 0.924926723861278
Train a XGBoost model
xgb loading...
xgboost accuracy on validation set:: 0.8450141442703957
Train a DecisionTree model
DecisionTree accuracy on validation set: 0.9529606412642719
Train a KNeighbors model
KNN accuracy on validation set: 0.9533813996277358
Train a RandomForest model
RandomForest accuracy on validation set: 0.9537379745120272
Train a SVM model
Load SVM model...
SVM accuracy on validation set: 0.9589510993203683
Load AdaBoost model...
AdaBoost accuracy on validation set: 0.9577886651975781
Train a LogisticRegression model
Load LogisticRegression model...
LogisticRegression accuracy on validation set: 0.9581880290679846
Train a MLP model
Load MLP model...
MLP accuracy on validation set: 0.9572395398757693
Train a GBDT model
Load GBDT model...
GBDT accuracy on validation set: 0.9574820107970875
Train a cbc model
Load cbc model...
CatBoost accuracy on validation set: 0.9566190995771022
Train a LGBM model
Load LGBM model...
LightGBM accuracy on validation set: 0.9589225733296249
```
## 1 环境补充安装
```
pip install catboost joblib
```
## 2 加压数据集
```
unzip dataset/ccf_offline_stage1_train.zip
#补充下载数据集，太大了我无法上传
wget https://github.com/yangchengjun/o2o_dataset/blob/master/dataset/ccf_online_stage1_train.zip -o dataset/
unzip dataset/ccf_online_stage1_train.zip 
```
## 3 运行训练与测试
```
python homework_o2o.py
```
## 4 输出说明

> output_files/baseline_importance.csv

为各个特征重要性得分，降序排列

> output_files/baseline.csv

4列分别为[用户id，优惠券id，收到优惠券日期，预测概率]

> output_files/models

存放训练好的模型

> output_files/figures

保存了精度对比图像

![fig](output_files/figures/auc_of_different_models.png 'fig')

## 模型保存功能
部分模型训练时间较长,如svm只用了10%训练集，所以创建了output_files/models文件夹保存训练好的模型，第一次训练后模型会保存，第二次运行时会检测是否保存有模型，如果想重新训练某个模型，直接删除保存的.pkl文件即可.
