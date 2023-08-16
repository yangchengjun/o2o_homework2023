

# 一、基础数学知识
补充第一点的基础数学知识，来理解第二、三点这些算法的设计原理，以及面对不同问题时，应该如何调整这些算法的参数。

# 1. 数学基础学习大纲

## 1.1 线性代数
### 1.1.1 向量、矩阵、张量的基本运算
- 加法、减法、数量乘法
- 矩阵的乘法、转置、逆

### 1.1.2 特征值与特征向量
- 特征分解的概念和性质
- 用于数据降维和其他线性变换

### 1.1.3 奇异值分解
- SVD的基本概念和应用
- 用于机器学习和信号处理

## 1.2 统计学与概率论
### 1.2.1 描述性统计
- 均值、中位数、方差、标准差和四分位数

### 1.2.2 条件概率
- 独立性、期望、方差、协方差

### 1.2.3 概率分布
- 连续（正态分布、指数分布）和离散（二项分布、泊松分布、迪利克雷分布）

### 1.2.4 贝叶斯规则

### 1.2.5 假设检验
- t检验、卡方检验

### 1.2.6 置信区间
- 如何为参数估计提供范围

### 1.2.7 相关性和因果性
- 如何评估变量之间的关系

## 1.3 信息论
### 1.3.1 熵 (Entropy)
- 描述信息的不确定性
- 计算单个变量的熵和多个变量的联合熵

### 1.3.2 互信息与相对熵 (KL散度)
- 描述两个随机变量之间的信息量
- 评估两个概率分布的相似度

### 1.3.3 最大熵原理
- 理解最大化熵的意义
- 应用于概率分布的估计

## 1.4 函数空间与函数逼近
### 1.4.1 希尔伯特空间
- 理解函数的内积、长度和正交性的概念
- 用于无限维度的函数空间

### 1.4.2 正交与标准正交基
- 构建函数的正交基
- 函数的Fourier级数表示

### 1.4.3 Fourier 和 Laplace 变换
- 理解变换的物理和数学意义
- 应用于信号处理和系统分析

### 1.4.4 泛函分析的基础
- 泛函的变分
- 了解泛函的极值问题

## 1.5 图论
### 1.5.1 图的基本概念与性质
- 定义点、边及其属性
- 不同类型的图：有向图、无向图、加权图等

### 1.5.2 树和最小生成树
- 理解树和森林的性质
- 用Prim和Kruskal算法寻找最小生成树

### 1.5.3 图的遍历
- 深度优先搜索和广度优先搜索的算法和应用
- 连通性的检验和应用

### 1.5.4 最短路径问题
- Dijkstra算法：单源最短路径
- Floyd-Warshall算法：多源最短路径



# 二、经典学习算法
## 2.1 基于概率的方法

### 2.1.1 高斯朴素贝叶斯 (Gaussian Naive Bayes)
高斯朴素贝叶斯是一个基于贝叶斯定理的分类算法，假设每个类中的特征都遵循高斯分布。
公式为:
$$ P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma^2_y}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma^2_y}\right) $$

### 2.1.2 多项式朴素贝叶斯 (Multinomial Naive Bayes)
多项式朴素贝叶斯常用于文本分类任务，其中特征表示各单词出现的次数或频率。

### 2.1.3 伯努利朴素贝叶斯 (Bernoulli Naive Bayes)
伯努利朴素贝叶斯适用于二值特征，如文本分类中的词袋模型。

## 2.2 基于距离的方法

### 2.2.1 KNN模型 (K-Nearest Neighbors)
KNN是一个基于实例的学习算法，分类一个点基于其k个最近邻的投票结果。
公式为:
$$ y = \frac{1}{k} \sum_{i=1}^{k} y_i $$

### 2.2.2 Radius Neighbors Classifier
该算法基于每个点的固定半径，而不是k个最近邻。

### 2.2.3 最近质心分类 (Nearest Centroid Classifier)
分类基于数据点到各类质心的距离。

## 2.3 基于决策的方法

### 2.3.1 决策树模型 (Decision Tree)
决策树是一个树形结构，每个节点代表一个特征，每个分支代表一个决策规则。

### 2.3.2 CART (Classification and Regression Trees)
CART可以用于分类和回归任务，它创建二叉树。

## 2.4 基于边界的方法

### 2.4.1 SVM模型 (Support Vector Machines)
SVM试图找到最大化间隔的超平面来分类数据。
公式为:
$$ w \cdot x + b = 0 $$

### 2.4.2 线性判别分析 (Linear Discriminant Analysis, LDA)
LDA用于降维并找到最佳的投影，使得类之间的距离最大化。

## 2.5 基于优化的方法

### 2.5.1 逻辑回归模型 (Logistic Regression)
逻辑回归用于预测某一事件的概率。
公式为:
$$ P(Y=1|X) = \frac{1}{1 + e^{-(w \cdot x + b)}} $$

### 2.5.2 岭回归 (Ridge Regression)
岭回归是线性回归的一种正则化版本，添加了L2正则项。

### 2.5.3 套索回归 (Lasso Regression)
套索回归是线性回归的一种正则化版本，添加了L1正则项。

## 2.6 基于神经网络的方法

### 2.6.1 多层感知机模型 (Multi-Layer Perceptron, MLP)
MLP是一种前馈神经网络，包括输入层、一个或多个隐藏层和输出层。

### 2.6.2 卷积神经网络 (Convolutional Neural Networks, CNN)
CNN主要用于图像处理和识别。

### 2.6.3 循环神经网络 (Recurrent Neural Networks, RNN)
RNN用于处理时序数据或序列数据。

## 2.7 基于集成学习(Ensemble Learning)的方法
组合多个学习器（通常是决策树）结合在一起，以创建更强大的学习器。
### 2.7.1 随机森林 (Random Forest)
通过集成多个决策树来进行预测。
对于分类问题，给定一个输入样本$x$，随机森林的输出为:
$$ \hat{y} = \text{mode}(\hat{y}_1, \hat{y}_2, ..., \hat{y}_B) $$
其中，$\hat{y}_i$是第$i$棵树的输出，$B$是决策树的数量。
   
对于回归问题:
$$ \hat{y} = \frac{1}{B} \sum_{i=1}^B \hat{y}_i $$
其中，$\hat{y}_i$是第$i$棵树的输出。
### 2.7.2 AdaBoost (Adaptive Boosting)
是一个逐步添加弱学习者的算法，每次迭代会更加关注之前被错误分类的样本。
公式为:
$$ F(x) = \sum_{t=1}^{T} \alpha_t h_t(x) $$
其中，$\alpha_t$是弱学习者的权重，$h_t(x)$是弱学习者的预测。

### 2.7.3 GBDT (Gradient Boosting Decision Tree)
是一种梯度提升算法，使用决策树作为基学习器。
公式为:
$$ F(x) = \sum_{t=1}^{T} \gamma_t h_t(x) $$
其中，$\gamma_t$是决策树的权重。

### 2.7.4 XGBoost
XGBoost是一个优化过的分布式梯度增强库，旨在实现高效、灵活和便携。与传统的GBDT相比，它引入了正则化项来控制模型的复杂性，从而防止过拟合。其目标函数为：
$$ \text{obj}(\theta) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k) $$
其中，$l$是损失函数，$\Omega(f_k)$是每个模型$f_k$的复杂性度量。

### 2.7.5 CatBoost
CatBoost是一个基于梯度提升的机器学习算法，特别为分类特征的处理进行了优化。CatBoost对每个分类特征进行了数字化，避免了独热编码所带来的维度爆炸。此外，它使用了一种称为“有序提升”的策略，该策略在每一次迭代中都使用新的数据排列，以减少过拟合并增强模型的泛化能力。

### 2.7.6 LightGBM
LightGBM是一个基于梯度提升的决策树算法，与其他树增强方法相比，它采用了一种称为“直方图优化”的策略。在直方图优化中，连续的特征值被分桶到离散的区间中，这大大减少了计算的复杂性。此外，LightGBM还使用了带深度限制的“叶子增长”策略，而不是传统的“深度增长”策略，从而使其在处理大型数据集时更加高效。
公式与GBDT相似，但在计算和分割选择上有所不同，它更倾向于选择更少的分割，从而加速训练和减少过拟合。

### 2.7.7 Bagging
Bagging，也称为自举汇聚法，它通过从训练数据中随机选择样本（有放回）来生成多个训练集，并对每个训练集训练一个模型。最后，所有模型的预测结果会进行平均（回归）或投票（分类）以得到最终的预测。
假设有$B$个自举样本，对于任意一个测试样本：
$$ \hat{f}_{bagging}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}^{*b}(x) $$

### 2.7.8 Stacking
Stacking是一种集成方法，其中多个基模型首先在数据上进行训练，然后另一个模型（通常称为“元模型”）对这些基模型的输出进行训练，以产生最终的预测结果。此方法的关键在于如何选择和组合不同的基模型，以及如何设计元模型，使其能够从基模型的预测中捕获到有价值的信息。




# 三、调参优化相关算法
## 3.1 初始化方法
### 3.1.1 随机初始化 (Random Initialization)
随机初始化是从一个随机分布（例如均匀分布或正态分布）中选择权值的方法。例如:
$$ w \sim U(-\epsilon, \epsilon) $$

### 3.1.2 零初始化 (Zero Initialization)
这种方法是简单地将所有权值初始化为0：
$$ w = 0 $$

### 3.1.3 Xavier/Glorot 初始化
为Sigmoid和tanh激活函数设计的初始化方法。权值从以下分布中选择：
$$ w \sim \mathcal{N}\left(0, \frac{1}{n}\right) $$
其中 $n$ 是输入单元的数量。

### 3.1.4 He初始化
为ReLU激活函数及其变种设计的初始化方法。权重从以下分布中选择：
$$ w \sim \mathcal{N}\left(0, \frac{2}{n}\right) $$
其中 $n$ 是输入单元的数量。

### 3.1.5 LeCun初始化
为Sigmoid激活函数设计的初始化方法。权值从以下分布中选择：
$$ w \sim \mathcal{N}\left(0, \frac{1}{\sqrt{n}}\right) $$
其中 $n$ 是输入单元的数量。

### 3.1.6 正态分布初始化 (Normal Initialization)
权值从标准正态分布中选择：
$$ w \sim \mathcal{N}(0, \sigma^2) $$

### 3.1.7 均匀分布初始化 (Uniform Initialization)
权值从均匀分布中选择。例如，可以选择从区间 $[-a, a]$ 的均匀分布：
$$ w \sim U(-a, a) $$

### 3.1.8 正交初始化 (Orthogonal Initialization)
正交初始化是将权重矩阵初始化为正交矩阵。对于一个给定的2D权重矩阵 $W$ (例如在全连接层或RNNs中)，其要求是 $W^T W = I$，其中 $I$ 是单位矩阵。具体的初始化步骤通常如下：
1. 生成一个随机矩阵 $R$，其元素是从标准正态分布中随机选择的。
2. 使用Singular Value Decomposition (SVD) 对 $R$ 进行分解: 
$$ R = U \Sigma V^T $$
3. 选择 $U$ (对于 $R$ 的行数 ≥ 列数) 或 $V$ (对于 $R$ 的行数 < 列数) 作为正交初始化的权重矩阵。

### 3.1.9 稀疏初始化 (Sparse Initialization)
稀疏初始化的目标是创建一个大部分元素为零的权值矩阵，从而确保初始权值矩阵是稀疏的。具体的初始化步骤可以如下：
1. 对于权重矩阵 $W$ 的每一列，随机选择 $s$ 个元素，其中 $s$ << $W$ 的行数。
2. 这 $s$ 个元素从某种分布（例如正态分布）中随机选择： 
$$ w_{ij} \sim \mathcal{N}(0, \sigma^2) $$
其中 $i$ 是随机选择的行索引，$j$ 是列索引。
3. 将 $W$ 矩阵中的所有其他元素设置为0。


### 3.1.10 通过预训练的模型初始化 (Initialization from a pre-trained model)
这是一种迁移学习方法，其中一个预先训练好的模型的权值被用作新模型的初始化。这可以帮助新模型快速收敛，特别是当新任务与预训练模型的任务相关时。

## 3.2 损失函数
### 3.2.1 均方误差 (MSE)
均方误差是回归问题中常用的损失函数。其计算公式为:
$$ L(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

### 3.2.2 交叉熵损失 (Cross Entropy Loss)
交叉熵损失常用于分类问题。其计算公式为:
$$ L(y, p) = - \sum_{i} y_i \log(p_i) $$

### 3.2.3 合页损失 (Hinge Loss)
合页损失常用于支持向量机和一些分类问题。其计算公式为:
$$ L(y, f(x)) = \max(0, 1 - y \cdot f(x)) $$

### 3.2.4 Huber 损失
Huber损失是MSE和MAE之间的折衷。其计算公式为:
$$ L(y, \hat{y}) = 
\begin{cases} 
0.5(y - \hat{y})^2 & \text{for } |y - \hat{y}| \leq \delta \\
\delta |y - \hat{y}| - 0.5 \delta^2 & \text{otherwise}
\end{cases}
$$

### 3.2.5 对数损失 (Log Loss)
对数损失是二分类问题中交叉熵损失的特例。其计算公式为:
$$ L(y, p) = -y \log(p) - (1 - y) \log(1 - p) $$

### 3.2.6 Softmax 损失
Softmax损失是多分类问题中交叉熵损失的扩展。公式与交叉熵相似，但包括了Softmax函数。

### 3.2.7 余弦相似性损失 (Cosine Similarity Loss)
这种损失衡量两个向量之间的余弦相似性。其计算公式为:
$$ L(a, b) = 1 - \frac{a \cdot b}{\|a\|_2 \|b\|_2} $$

### 3.2.8 KL 散度 (Kullback-Leibler Divergence)
KL散度衡量两个概率分布的差异。其计算公式为:
$$ D_{KL}(p||q) = \sum_{i} p(i) \log\left(\frac{p(i)}{q(i)}\right) $$

### 3.2.9 均方对数误差 (Mean Squared Logarithmic Error)
此损失函数对预测的和真实的值取对数后计算均方误差。公式为:
$$ L(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} (\log(1 + y_i) - \log(1 + \hat{y}_i))^2 $$

### 3.2.10 总变差损失 (Total Variation Loss)
此损失用于图像处理中，鼓励图像的空间连续性。公式通常涉及图像的相邻像素。

### 3.2.11 三元组损失 (Triplet Margin Loss)
三元组损失用于学习相似性。其计算公式为:
$$ L(a, p, n) = \max(0, \|a - p\|_2^2 - \|a - n\|_2^2 + \text{margin}) $$

### 3.2.12 Contrastive 损失 (Contrastive Loss)
Contrastive损失用于学习相似性。其计算公式为:
$$ L(y, d) = \frac{1}{2} y d^2 + \frac{1}{2} (1 - y) \max(0, margin - d)^2 $$
其中$d$是两个实例之间的距离。

### 3.2.13 聚焦损失 (Focal Loss)
Focal损失是为了解决类别不平衡设计的。其调整了交叉熵损失，使模型更关注难分类的样本。公式为:
$$ L(y, p) = -\alpha y (1-p)^\gamma \log(p) - (1 - y) p^\gamma \log(1 - p) $$
其中$\alpha$和$\gamma$是超参数。

### 3.2.14 Dice 损失 (Dice Loss)
Dice损失常用于图像分割任务。其衡量两个样本集合的相似性。公式为:
$$ L(y, \hat{y}) = 1 - \frac{2 \sum_{i=1}^{n} y_i \hat{y}_i}{\sum_{i=1}^{n} y_i + \sum_{i=1}^{n} \hat{y}_i} $$

### 3.2.15 Categorical Hinge 损失
这是多分类的合页损失。其计算公式为:
$$ L(y, f(x)) = \max(0, 1 + \max_{j \neq y} f_j(x) - f_y(x)) $$

### 3.2.16 Poisson 损失 (Poisson Loss)
Poisson损失用于计数问题。其计算公式为:
$$ L(y, \hat{y}) = \hat{y} - y \log(\hat{y}) $$

### 3.2.17 Log-cosh 损失 (Log-cosh Loss)
Log-cosh是另一个用于回归任务的损失函数。其计算公式为:
$$ L(y, \hat{y}) = \log(\cosh(\hat{y} - y)) $$



## 3.3 优化方法
### 3.3.1 SGD (随机梯度下降)
SGD是最基本的优化算法。其更新公式为：
$$ w_{t+1} = w_t - \eta \nabla J(w_t) $$
其中$\eta$是学习率，$\nabla J(w_t)$是损失函数$J$在时间$t$的梯度。

### 3.3.2 SGDM（sgd with momentum）
SGDM优化器在SGD的基础上加入了动量项，使参数更新具有“惯性”效果。其更新公式为：
$$ v_{t+1} = \beta v_t + (1-\beta) \nabla J(w_t) $$
$$ w_{t+1} = w_t - \eta v_{t+1} $$
其中$v_t$是动量项。

### 3.3.3 Adagrad
Adagrad根据参数的历史梯度来调整学习率。其更新公式为：
$$ G_t = G_{t-1} + \nabla J(w_t) \odot \nabla J(w_t) $$
$$ w_{t+1} = w_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla J(w_t) $$
其中$G_t$是历史梯度的累加，$\odot$表示逐元素乘法。

### 3.3.4 RMSprop
RMSprop是为了解决Adagrad学习率递减太快的问题而提出的。其更新公式为：
$$ E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta) g_t^2 $$
$$ w_{t+1} = w_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t $$
其中$g_t$是梯度。

### 3.3.5 Adam
Adam结合了Momentum和RMSprop的思想。其更新公式为：
$$ m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t $$
$$ v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 $$
$$ \hat{m}_t = \frac{m_t}{1-\beta_1^t} $$
$$ \hat{v}_t = \frac{v_t}{1-\beta_2^t} $$
$$ w_{t+1} = w_t - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$

### 3.3.6 Adadelta
Adadelta是RMSprop的一个扩展，不需要设置默认的学习率。它基于RMSprop的公式，但是为了减少其学习率的急剧下降，它使用了均方根误差。更新公式如下：
$$ E[g^2]_t = \rho E[g^2]_{t-1} + (1-\rho) g_t^2 $$
$$ \Delta w_t = - \frac{\sqrt{E[\Delta w^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} g_t $$
$$ E[\Delta w^2]_t = \rho E[\Delta w^2]_{t-1} + (1-\rho) \Delta w_t^2 $$
$$ w_{t+1} = w_t + \Delta w_t $$
其中$\rho$是一个介于0和1之间的衰减因子。

### 3.3.7 AdamW
AdamW是Adam的变种，引入了权重衰减，其目的是在优化过程中直接将权重衰减纳入权重更新步骤。其更新公式与Adam相似，但具有权重衰减：
$$ w_{t+1} = (1 - \eta \lambda) w_t - \eta \hat{m}_t/(\sqrt{\hat{v}_t} + \epsilon) $$
其中$\lambda$是权重衰减系数。

### 3.3.8 Nadam
Nadam结合了Adam和Nesterov动量的优点。其更新公式为：
$$ m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t $$
$$ v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 $$
$$ \hat{m}_t = \frac{m_t}{1-\beta_1^t} $$
$$ \hat{v}_t = \frac{v_t}{1-\beta_2^t} $$
$$ w_{t+1} = w_t - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \frac{\eta \beta_1 \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$

### 3.3.9 FTRL (Follow The Regularized Leader)
FTRL适用于大规模的稀疏数据，公式相对复杂，与特定的正则化形式关联。它的主要特点是在线学习和L1正则化。其更新公式依赖于选择的正则化策略。

### 3.3.10 LBFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)
LBFGS是一个迭代的方法，用于非线性优化问题。它是二阶方法，但不直接计算Hessian矩阵或其逆矩阵，而是近似Hessian矩阵。由于它的详细更新过程涉及到许多矩阵运算和条件判断，这里不直接给出具体的公式。



