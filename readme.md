# o2o_dataset
## 1 环境补充安装
```
pip install catboost joblib
```
## 2 加压数据集
```
unzip dataset/ccf_offline_stage1_train.zip
unzip dataset/ccf_online_stage1_train.zip
```
## 3 运行训练与测试
```
python 2023homework_o2o.py
```
## 4 输出说明

> sciclass/output_files/baseline_importance.csv

为各个特征重要性得分，降序排列

> sciclass/Baseline_o2o.py

4列分别为[用户id，优惠券id，收到优惠券日期，预测概率]

> sciclass/output_files/models

存放训练好的模型

> sciclass/output_files/figures

保存了精度对比图像

![fig](output_files/figures/auc_of_different_models.png 'fig')

## 模型保存功能
部分模型训练时间较长,如svm只用了10%训练集，所以创建了output_files/models文件夹保存训练好的模型，第一次训练后模型会保存，第二次运行时会检测是否保存有模型，如果想重新训练某个模型，直接删除保存的.pkl文件即可.
