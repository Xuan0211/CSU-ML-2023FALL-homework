# Ensemble Learning

集成学习的思想很简单，就是构建多个学习器一起结合来完成具体的学习任务。通过将多个学习器进行结合，常可获得比单一学习器显著优越的泛化性能，对弱学习器尤为明显。（三个臭皮匠，顶个诸葛亮）

弱学习器是指学习器的学习正确率仅比随机猜测略好，强学习器是指学习器的学习正确率很高。集成学习就是结合多个弱分类器组成一个强分类器。

集成学习可以分成两类：

- 个体学习器间存在强依赖关系，必须串行生成学习模型。代表：Boosting（AdaBoost, Gradient Boosting Machine）。
- 个体学习器间不存在强依赖关系，可同时生成学习模型。代表：Bagging和随机森林（Random Forest）。

AdaBoost通过改变训练数据的权重分布来训练一组弱分类器，把这些弱分类器线性组合成为一个强分类器。GBDT结合提升树模型和梯度提升的优点，使用新弱分类器拟合前一次迭代模型的样本余量，逐渐降低训练误差。Bagging和随机森林利用自助采样采集T组训练样本集，分别训练T个分类器，对T个分类器的预测结果进行投票决定模型的最终预测结果。
本次作业我们要实现adaboost, GBDT和Random Forest三个模型，为方便模型编写，我们采用scikit-learn构建CART树作为弱学习器。此外，添加了一个实战机器学习任务，Kaggle经典竞赛Titanic生存预测，通过这个任务大家可以比较不同集成学习算法的优劣。

提示：scikit-learn是一个简单而有效的python机器学习算法库，里面包含了许多常见的机器学习算法（包括本课程讲的算法）。这里直接使用scikit-learn实现的CART算法，方便我们完成实验。点击[这里](http://scikit-learn.org/stable/)查看scikit-learn的官方文档，点击[这里](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)查看scikit-learn实现的CART算法的API接口。


```python
# 导入所需的包
import numpy as np
import pandas as pd 
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.datasets import make_circles
from sklearn.datasets import make_classification
```

####  **任务1：**
构建一个简单的二分类数据集，了解并使用scikit-learn的DecisionTreeClassifier模块快速构建CART树并拟合数据集。

请参考scikit-learn官方文档
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier


```python
# 使用scikit-learn生成一个简单的二分类数据集
X, y = make_circles(n_samples=100, noise=0.5, factor=0.2, random_state=1)

# 二分类标签一般是‘0’和‘1’，adaboost算法的标签为‘1’，‘-1’，修改adaboost标签‘0’变为‘-1’
y_ada = y.copy()
y_ada[y_ada == 0] = -1

# 可视化生成的数据集
plt.scatter(X[:, 0], X[:, 1], c=y)
```




    <matplotlib.collections.PathCollection at 0x1d1dbd8ef48>




    
![png](ML2023_HW7_EnsembleLearning_files/ML2023_HW7_EnsembleLearning_3_1.png)
    



```python
# 首先使用CART对数据进行训练

# 要求采用DecisionTreeClassifier构建最大深度为5，其余为默认参数的决策树模型，并使用fit方法拟合生成的(X, y)数据，采用score方法测试模型精度
cart_model = DecisionTreeClassifier(max_depth = 5)
cart_model.fit(X, y)

# 可视化CART的分类效果
x1_grid = np.linspace(-2, 2, 100)
x2_grid = np.linspace(-2, 2, 100)
x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
y_grid_simple = np.zeros_like(x1_grid)
X_grid = np.hstack([x1_grid.reshape(-1, 1), x2_grid.reshape(-1, 1)])

# 采用predict方法使用生成模型进行分类预测
y_grid = cart_model.predict(X_grid, check_input=True)
y_grid_simple = y_grid.reshape(100, 100)

#可视化结果
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.contourf(x1_grid, x2_grid, y_grid_simple, alpha=0.3)
```




    <matplotlib.contour.QuadContourSet at 0x1d1dd232c88>




    
![png](ML2023_HW7_EnsembleLearning_files/ML2023_HW7_EnsembleLearning_4_1.png)
    


## AdaBoost

提升方法是从弱学习算法出发，反复学习，得到一系列弱分类器，然后组合这些弱分类器，构成一个强分类器。对于提升方法来说，有两个问题需要回答：一是在每一轮如何改变训练数据的权值或概率分布；二是如何将弱分类器组合成一个强分类器。关于第1个问题，AdaBoost的做法是，提高那些被前一轮弱分类器错误分类样本的权值，而降低那些被正确分类样本的权值。这样一来，那些没有得到正确分类的数据，由于其权值的加大而受到后一轮的弱分类器的更大关注。于是，分类问题被一系列的弱分类器“分而治之”。至于第2个问题，即弱分类器的组合，AdaBoost采取加权多数表决的方法。具体地，加大分类误差率小的弱分类器的权值，使其在表决中起较大的作用，减少分类误差率大的弱分类器的权值，使其在表决中起较少的作用。

假设给定一个二分类的训练数据集
$$
T = \{(x_1, y_1), (x_2, y_2), \cdots, (x_N, y_N)\}
$$
其中，每个样本点由实例与标记组成。实例$x_i \in \mathcal{X} \subseteq R^n$，$y_i \in \mathcal{Y} = \{-1, +1\}$，$\mathcal{X}$是实例空间，$\mathcal{Y}$是标记集合。AdaBoost利用以下算法，从训练数据中学习一系列弱分类器或基本分类器，并将这些弱分类器线性组合成为一个强分类器。

输入：训练数据集$T=\{(x_1, y_1), (x_2, y_2), \cdots, (x_N, y_N)\}$，其中$x_i \in \mathcal{X} \subseteq R^n$，$y_i \in \mathcal{Y} = \{-1, +1\}$；  
输出：最终分类器$G(x)$。  
(1)初始化训练数据的权值分布
$$
D_1 = (w_{11}, \cdots, w_{1i}, \cdots, w_{1N}), w_{1i} = \frac{1}{N}, i = 1, 2, \cdots, N
$$
(2)对$m=1,2,\cdots,M$  
(a)使用具有权值分布$D_m$的训练数据集学习，得到基本分类器
$$
G_m(x):\mathcal{X}\to \{-1, +1\}
$$
(b)计算$G_m(x)$在训练数据集上的分类误差率
$$
e_m = \sum_{i=1}^{N} P(G_m(x_i) \not = y_i) = \sum_{i=1}^{N} w_{mi} I(G_m(x) \not = y_i)
$$
(c)计算$G_m(x)$的系数
$$
\alpha_m = \frac{1}{2} \log \frac{1-e_m}{e_m} \tag{1}
$$
(d)更新训练数据集的权值分布
$$
D_{m+1} = (w_{m+1,1}, \cdots, w_{m+1,i}, \cdots, w_{m+1,N})
$$
$$
w_{m+1, i} = \frac{w_{mi}}{Z_m} \exp (-\alpha_m y_i G_m(x_i))), i = 1, 2, \cdots, N \tag{2}
$$
这里，$Z_m$是规范化因子
$$
Z_m = \sum_{i=1}^{N} w_{mi} \exp(-\alpha_m y_i G_m(x_i))
$$
它使$D_{m+1}$成为一个概率分布。  
(3)构建基本分类器的线性组合
$$
f(x) = \sum_{m=1}^{M} \alpha_m G_m(x)
$$
得到最终分类器
$$
G(x) = \mathrm{sign}(f(x)) = \mathrm{sign}\left( \sum_{m=1}^{M} \alpha_m G_m(x) \right)
$$
到此，算法结束。

观察公式（1）$e_m$-$\alpha_m$的曲线图

<div align=center>
<img src="./images/image1.png" >
<br>
图1. e_m alpha_m曲线图
</div>

可以看到，当分类误差率低时，$\alpha_m$的值较高，当分类误差率高时，$\alpha_m$的值较低。

来考虑一下，$e_m$是否会取0或1。如果$e_m = 0$，说明弱分类器的效果非常好，正确率100%，可以停止迭代了！如果$e_m = 1$，说明弱分类器的对所有样本都分类错误，学习出一个效果最差的分类器， 但这种情况出现的概率微乎其微！

训练数据的权值分布的更新公式（2）可以写成：
$$
w_{m+1,i} = \begin{cases} \frac{w_{mi}}{Z_m}e^{-\alpha_m}, & G_m(x_i) = y_i \\ \frac{w_{mi}}{Z_m}e^{\alpha_m}, & G_m(x_i) \not = y_i \end{cases}
$$
由此可知，被基本分类器$G_m(x)$误分类样本的权值得以扩大，而被正确分类样本的权值得以缩小。因此，误分类样本在下一轮学习中起更大的作用。不改变所给的训练数据，而不断改变训练数据权值的分布，使得训练数据在基本分类器的学习中起不同的作用，这是AdaBoost的一个特点。

#### **任务2：**
根据上述描述构建adaboost算法

弱学习器采用scikit-learn的DecisionTreeClassifier实现，其中决策树的深度可由输入参数控制


```python
def adaboost(X, y, M, max_depth=5):
    """
    adaboost函数，使用CART作为弱分类器
    参数:
        X: 训练样本
        y: 样本标签, y = {-1, +1}
        M: 使用M个弱分类器
        max_depth: 基学习器CART决策树的最大深度
    返回:
        F: 生成的模型
    """
    num_X, num_feature = X.shape
    
    ### START CODE HERE ###
    # 初始化训练数据的权值分布
    # 初始化D为一个概率分布，其中每个元素为1/num_X
    D = [1/num_X] * num_X
    G = []
    alpha = []
    for m in range(M):
        # 使用具有权值分布D_m的训练数据集学习，得到基本分类器
        # 使用DecisionTreeClassifier，设置树深度为max_depth
        G_m = DecisionTreeClassifier(max_depth = max_depth)
        # 开始训练
        G_m.fit(X, y)
        # 计算G_m在训练数据集上的分类误差率
        y_pred = G_m.predict(X)
        e_m = 0
        for i in range(num_X):
            if y_pred[i] != y[i]:
                e_m += D[i]
        print(e_m)
        if e_m == 0:
            break
        if e_m == 1:
            raise ValueError("e_m = {}".format(e_m))
        # 计算G_m的系数        
        alpha_m = 0.5 * np.log((1-e_m)/e_m)
        # 更新训练数据集的权值分布
        for i in range(num_X):
            D[i] = D[i] * np.exp(-alpha_m * y[i] * y_pred[i])
        D = D/np.sum(D)
        # 保存G_m和其系数
        G.append(G_m)
        alpha.append(alpha_m)
    
    # 构建基本分类器的线性组合
    def F(X):
        num_G = len(G)
        score = 0
        for i in range(num_G):
            score +=  alpha[i] * G[i].predict(X)
        return np.sign(score)
        
    ### END CODE HERE ###
    return F
```

## Gradient Boosting Machine (GBM)

GBM和AdaBoost一样采用加法模型：$H(x) = \sum_{t=1}^{T} \alpha_t h_t(x)$，但GBM拓展为可以采用其他任意损失$l$（如前面介绍过的平方损失、交叉熵损失等）。

GBM一般采用决策树（或回归树）作为基学习器，称为Gradient Boosting Decision Tree (GBDT)，针对不同问题使用不同的损失函数，分类问题使用指数损失函数，回归问题使用平方误差损失函数。

GBDT的加法模型为:
$$
f_m(x) = \sum_{m=1}^{M} T(x;\Theta_m)
$$
其中$T(x;\Theta_m)$表示决策树；$\Theta_m$为决策树参数；M为树的个数。

GBDT采用前向分步算法。首先确定初始提升树$f_0(x) = 0$，第m步的模型是
$$
f_m(x) = f_{m-1}(x) + T(x; \Theta_m)
$$
其中，$f_{m-1}$为当前模型，通过经验风险极小化确定下一棵决策树的参数$\Theta_m$，
$$
\hat{\Theta}_m = \arg \underset{\Theta_m}{\min} \sum_{i=1}^{N} L(y_i, f_{m-1}(x_i) + T(x_i; \Theta_m))
$$
为了能够得到最优的下一棵决策树，Freidman提出了梯度提升（gradient boosting）算法。这是利用最速下降法的近似方法，其关键是利用损失函数的负梯度在当前模型的值
$$
-\left[ \frac{\partial L(y, f(x_i))}{\partial f(x_i)} \right]_{ f(x) = f_{m-1}(x) }
$$
作为回归问题提升树算法中的残差的近似值，拟合一个回归树。

GBDT算法  
输入：训练数据集$T=\{(x_1, y_1), (x_2, y_2), \cdots, (x_N, y_N)\}$，$x_i \in \mathcal{X} \subseteq R^n$，$y_i \in \mathcal{Y} \subseteq R$;  
输出： 回归树$\hat{y}(x)$。  
（1）初始化
$$
f_0(x) = \arg \underset{c}{\min}\sum_{i=1}^{N}L(y_i, c)
$$
（2）对$m=1,2,\cdots,M$  
  (a)对$i=1,2,\cdots,N$，计算
$$
r_{mi} = -\left[ \frac{\partial L(y, f(x_i))}{\partial f(x_i)} \right]_{f(x) = f_{m-1}(x)}
$$
  (b)对$r_{mi}$拟合一个回归树，得到第m棵树的叶结点区域$R_{mj}$，$j=1,2,\cdots,J$  
  (c)对$j=1,2,\cdots,J$，计算
$$
c_{mj} = \arg \underset{c}{\min} \sum_{x_i \in R_{mj}} L(y_i, f_{m-1}(x_i) + c)
$$
  (d)更新$f_m(x) = f_{m-1}(x) + \sum_{j=1}^{J} c_{mj}I(x \in R_{mj})$  
（3）得到回归树
$$
\hat{f}(x) = f_M(x) = \sum_{m=1}^{M} \sum_{j=1}^{J}c_{mj}I(x \in R_{mj})
$$

虽然说GBDT使用的是回归树，当是也可以用于分类问题，还记得Logistic Regression吗？逻辑回归解决的是二元分类问题，softmax可以解决多分类问题。如果损失函数$L(y, f(x_i))$为交叉熵损失，GBDT就可以解决分类问题，如果损失函数$L(y, f(x_i))$为平方差损失，GBDT就可以解决回归问题。

$x_i$的平方差损失为
$$
L(y, f(x_i)) = \frac{1}{2} (y_i - f(x_i))^2
$$
对应的$r_{mi}$为
$$
r_{mi} = -\left[ \frac{\partial L(y, f(x_i))}{\partial f(x_i)} \right]_{f(x) = f_{m-1}(x)} = y - f_{m-1}(x)
$$

$x_i$的交叉熵损失为
$$
L(y, g(x_i)) = - y_i \log (g(x_i)) - (1 - y_i) \log (1 - g(x_i)) 
$$
其中$g(x_i) = sigmoid(f(x_i))$，
对应的$r_{mi}$为
$$
r_{mi} = -\left[ \frac{\partial L(y, f(x_i))}{\partial f(x_i)} \right]_{f(x) = f_{m-1}(x)} = y - g(f_{m-1}(x))
$$

#### **任务3：**
根据上述描述构建GBDT算法

**注意：**
GBDT的弱学习器应采用回归模型，弱学习器采用scikit-learn的DecisionTreeRegressor实现，具体使用和之前的分类模型类似，可以参考[这里](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor)，
其中决策树的深度可由输入参数控制


```python
from sklearn.tree import DecisionTreeRegressor
def sigmoid(x):
    """
    计算sigmoid函数值
    """
    return 1 / (1 + np.exp(-x))


def gbdt_classifier(X, y, M, max_depth=None):
    """
    用于分类的GBDT函数
    参数:
        X: 训练样本
        y: 样本标签，y = {0, +1}
        M: 使用M个回归树
        max_depth: 基学习器CART决策树的最大深度
    返回:
        F: 生成的模型
    """
    ### START CODE HERE ###
    # 用0初始化y_reg
    y_reg = np.zeros(len(y))
    f = []
    
    for m in range(M):
        # 计算r
        r = y - sigmoid(y_reg)
        # 拟合
        # 使用DecisionTreeRegressor，设置树深度为5，random_state=0
        f_m = DecisionTreeRegressor(max_depth=5, random_state=0)
        # 开始训练
        f_m.fit(X, r)
        # 更新f
        f.append(f_m)
        y_reg += f_m.predict(X)
    
    def F(X):
        num_X, _ = X.shape
        reg = np.zeros((num_X))
        for t in f:
            reg = reg + t.predict(X)
        y_pred_gbdt = reg
        # 以0.5为阈值，得到最终分类结果0或1
        one_position = y_pred_gbdt>=0.5
        y_pred_gbdt[one_position] = 1
        y_pred_gbdt[~one_position] = 0
        return y_pred_gbdt
    
    ### END CODE HERE ###
    return F
```

接下来，采用任务1实现CART的树策略作为弱学习器，即控制弱学习器CART树最大深度为5，分别构建adaboost和GBDT集成学习模型，弱学习器数目设置为10，查看最终结果。运行以下代码，可以看到adaboost和GBDT模型相较于一个CART决策树，准确率都得到了较大的提升。


```python
# 用adaboost和GBDT模型进行训练

adaboost_model = adaboost(X, y_ada, 10, max_depth=5)
gbdt_model = gbdt_classifier(X, y, 10, max_depth=5)

y_pre_ada = adaboost_model(X)
y_pre_gbdt = gbdt_model(X)

accuracy_ada = np.mean(y_pre_ada == y_ada)
accuracy_gbdt = np.mean(y_pre_gbdt == y)

print("Accuracy of Adaboost model is {}".format(accuracy_ada))
print("Accuracy of GBDT model is {}".format(accuracy_gbdt))

# 可视化CART的分类效果
x1_grid = np.linspace(-2, 2, 100)
x2_grid = np.linspace(-2, 2, 100)

x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)
X_grid = np.hstack([x1_grid.reshape(-1, 1), x2_grid.reshape(-1, 1)])

y_grid_ada = adaboost_model(X_grid)
y_grid_gbdt = gbdt_model(X_grid)

y_grid_simple_ada = y_grid_ada.reshape(100, 100)
y_grid_simple_gbdt = y_grid_gbdt.reshape(100, 100)

ada_fig=plt.figure()
ax1=ada_fig.add_subplot(111)
ax1.set_title('adaboost_classifier')
ax1.scatter(X[:, 0], X[:, 1], c=y_ada)
ax1.contourf(x1_grid, x2_grid, y_grid_simple_ada, alpha=0.3)

gbdt_fig=plt.figure()
ax2=gbdt_fig.add_subplot(111)
ax2.set_title('gbdt_classifier')
ax2.scatter(X[:, 0], X[:, 1], c=y)
ax2.contourf(x1_grid, x2_grid, y_grid_simple_gbdt, alpha=0.3)

plt.show()
```

    0.19000000000000003
    0.49999999999999967
    0.5000000000000002
    0.4999999999999998
    0.5000000000000002
    0.4999999999999998
    0.5000000000000002
    0.4999999999999998
    0.5000000000000002
    0.4999999999999998
    Accuracy of Adaboost model is 0.81
    Accuracy of GBDT model is 0.97
    


    
![png](ML2023_HW7_EnsembleLearning_files/ML2023_HW7_EnsembleLearning_12_1.png)
    



    
![png](ML2023_HW7_EnsembleLearning_files/ML2023_HW7_EnsembleLearning_12_2.png)
    


## Bagging

Bagging算法很简单，利用自助采样（有放回的均匀抽样）得到T组训练样本集，分别利用这些训练样本集训练T个分类器（CART or SVM or others），最后进行投票集成。

从偏差-方差分解的角度看，Bagging主要关注降低方差，因此它在不剪枝决策树、神经网络等易受样本扰动的学习器上效果更为明显。

## Random Forest

随机森林是Bagging的一个扩展变体，它充分利用“随机”的思想来增加各分类器的多样性。“随机”体现在两个方面：基于自助采样法来选择训练样本和随机选择特征（或属性）。随机选择特征是指，对基决策树的每个节点，先从该节点的属性集合中随机选择一个包含k个属性的子集，然后再从这个子集中选择一个最优属性用于划分。这里的参数k控制了随机性和引入程度，一般情况下，推荐值$k=\log_2d$(假设有d个属性)。随机森林的弱分类器一般是CART。随机森林的特点是可高度并行化、继承了CART的优点和克服了完全生长树的缺点。

Scikit-learn实现的CART算法默认随机选择特征，因此，直接采用bagging算法集成CART树就是Random Forest的实现函数。


#### **任务4：**
构建bagging算法

采用投票法作为最终集成的方法


```python
from sklearn import tree
def bagging(X, y, T, size, seed=0, max_depth=None):
    """
    Bagging算法，分类器为CART，用于二分类
    参数：
        X: 训练集
        y: 样本标签
        T: T组
        size: 每组训练集的大小
        seed: 随机种子
        max_depth: 基学习器CART决策树的最大深度
    返回：
        F: 生成的模型
    """
    classifiers = []
    m, n = X.shape
    
    ### START CODE HERE ###
    np.random.seed(seed)
    for i in range(T):
        # 使用np.random.choice选择size个序号，注意replace参数的设置，以满足有放回的均匀抽样。
        # Whether the sample is with or without replacement. 
        # Default is True, meaning that a value of a can be selected multiple times.
        index = np.random.choice(m, size, replace=True)
        X_group = [X[idx] for idx in index]
        y_group = [y[idx] for idx in index]
        # 使用tree.DecisionTreeClassifier，设置max_depth=max_depth, min_samples_split=2(生成完全树),random_state=0
        t = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=2, random_state=0)
        # 开始训练
        t.fit(X_group,y_group)
        classifiers.append(t)
        
    def F(X):
        # 计算所有分类器的预测结果
        result = np.zeros((X.shape[0], T))
        for i in range(T):
            result[:, i] = classifiers[i].predict(X)
        # 把预测结果组成 num_X * T 的矩阵
        result = np.array(result)
        # 计算"0"有多少投票
        vote_0 = np.sum(result == 0, axis=1)
        # 计算"1"有多少投票
        vote_1 = np.sum(result == 1, axis=1)
        # 选择投票数最多的一个标签
        pred = np.where(vote_0 > vote_1, 0, 1)
        return pred    
    ### END CODE HERE ###
    return F
```

至此，三大集成学习算法都已经实现了。集成学习这一章节结束，我们已经学习了大部分的经典的监督学习算法。下面我们考虑完成一个基础的机器学习应用。

## Titanic: Machine Learning from Disaster

Titanic生存预测是kaggle竞赛机器学习入门的经典题目，Kaggle提供的数据集中，共有1309名乘客数据，其中891是已知存活情况，剩下418则是需要进行分析预测的。我们采用其提供的训练数据titanic_train.csv来进行本次实验。

首先来看一下titanic_train.csv共有891条数据，包含一下内容：

PassengerId: 乘客编号  
Survived   : 存活情况（存活：1 ; 死亡：0）  
Pclass     : 客舱等级  
Name       : 乘客姓名  
Sex        : 性别  
Age        : 年龄  
SibSp      : 同乘的兄弟姐妹/配偶数  
Parch      : 同乘的父母/小孩数  
Ticket     : 船票编号  
Fare       : 船票价格  
Cabin      : 客舱号  
Embarked   : 登船港口  

PassengerId :   891 non-null int64  
Survived    :   891 non-null int64  
Pclass      :   891 non-null int64  
Name        :   891 non-null object  
Sex         :   891 non-null object  
Age         :   714 non-null float64  
SibSp       :   891 non-null int64  
Parch       :   891 non-null int64  
Ticket      :   891 non-null object  
Fare        :   891 non-null float64  
Cabin       :   204 non-null object  
Embarked    :   889 non-null object  

在实际的机器学习任务中，数据处理应占整个项目相当大的比例，怎样从原始数据中心得到有效的信息，对于建模及其关键。Titanic数据集中有部分信息是缺失的，如何选取有效字段进行保留，如何补全缺失值，使我们在进行建模前需要考虑的问题。
这里提供一个Titanic数据集比较全面的数据分析过程，大家可以从中学习和了解机器学习数据清洗的方法。[链接](https://www.kaggle.com/startupsci/titanic-data-science-solutions)

#### **任务5：**
处理Titanic训练数据

采用pandas工具进行数据分析[这里](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)可以查看其官方文档。

本次作业我们采用较为简单的方式处理该数据集，去掉姓名，乘客编号和缺失值较多的客舱号三项；采用均值来填满缺失的年龄信息；将性别和登船港口三列转为离散值进行处理；合并SibSp和Parch得到总的家人数。


```python
data_train = pd.read_csv("./titanic/train.csv")

### START CODE HERE ###
# 采用mean()得到年龄均值，填补缺失信息
data_train["Age"].fillna(data_train["Age"].mean(), inplace=True)

# 采用pd.get_dummies得到离散数据
dummies_Embarked = pd.get_dummies(data_train["Embarked"], prefix="Embarked")
dummies_Sex = pd.get_dummies(data_train["Sex"], prefix="Sex")
dummies_Pclass = pd.get_dummies(data_train["Pclass"], prefix="Pclass")

### END CODE HERE ###
# 采用pd.oncat合并原始数据和生成的离散数据
df = pd.concat([data_train, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df['family_num'] = df['SibSp'].values + df['Parch'].values

# 得到最终的训练数据的字段
train_df = df.filter(regex='Survived|Age.*|family_num|Fare.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*')
# 显示最终处理完成的数据信息
print(train_df.info())

train_np = np.array(train_df)
print(train_np.shape)


# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

print(X.shape)
print(y.shape)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 13 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   Survived    891 non-null    int64  
     1   Pclass      891 non-null    int64  
     2   Age         891 non-null    float64
     3   Fare        891 non-null    float64
     4   Embarked_C  891 non-null    uint8  
     5   Embarked_Q  891 non-null    uint8  
     6   Embarked_S  891 non-null    uint8  
     7   Sex_female  891 non-null    uint8  
     8   Sex_male    891 non-null    uint8  
     9   Pclass_1    891 non-null    uint8  
     10  Pclass_2    891 non-null    uint8  
     11  Pclass_3    891 non-null    uint8  
     12  family_num  891 non-null    int64  
    dtypes: float64(2), int64(3), uint8(8)
    memory usage: 41.9 KB
    None
    (891, 13)
    (891, 12)
    (891,)
    

#### 验证试验

采用不同的模型进行训练，包括CART树、adaboost、GBDT和Random Forest算法，其中树的最大深度都固定为10。  



```python
n_splits = 5
kf = KFold(n_splits)
accuracy_train_ada, accuracy_val_ada = 0, 0
accuracy_train_gbdt, accuracy_val_gbdt = 0, 0
accuracy_train_rf, accuracy_val_rf = 0, 0
accuracy_train_CART, accuracy_val_CART = 0, 0

for train, test in kf.split(X):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    y_train_ada = y_train.copy()
    y_train_ada[y_train_ada == 0] = -1
    y_test_ada = y_test.copy()
    y_test_ada[y_test_ada == 0] = -1
    
    adaboost_model = adaboost(X_train, y_train_ada, 100, max_depth=10)
    gbdt_model = gbdt_classifier(X_train, y_train, 100, max_depth=10)
    randomforest_model = bagging(X_train, y_train, 100, int(X_train.shape[0]*0.4), max_depth=10)
    CART = DecisionTreeClassifier(max_depth=10)
    CART.fit(X_train, y_train)
    
    y_train_pre_ada = adaboost_model(X_train)
    y_test_pre_ada = adaboost_model(X_test)
    y_train_pre_gbdt = gbdt_model(X_train)
    y_test_pre_gbdt = gbdt_model(X_test)
    y_train_pre_rf = randomforest_model(X_train)
    y_test_pre_rf = randomforest_model(X_test)
 
    accuracy_train_CART += CART.score(X_train, y_train)
    accuracy_val_CART += CART.score(X_test, y_test)
    accuracy_train_ada += np.mean(y_train_pre_ada == y_train_ada)
    accuracy_val_ada += np.mean(y_test_pre_ada == y_test_ada)
    accuracy_train_gbdt += np.mean(y_train_pre_gbdt == y_train)
    accuracy_val_gbdt += np.mean(y_test_pre_gbdt == y_test)
    accuracy_train_rf += np.mean(y_train_pre_rf == y_train)
    accuracy_val_rf += np.mean(y_test_pre_rf == y_test)
    
    
print("Accuracy of cart model in trainset is {}, in validation set is {}".format(accuracy_train_CART/n_splits, accuracy_val_CART/n_splits))
print("Accuracy of adaboost model in trainset is {}, in validation set is {}".format(accuracy_train_ada/n_splits, accuracy_val_ada/n_splits))
print("Accuracy of gbdt model in trainset is {}, in validation set is {}".format(accuracy_train_gbdt/n_splits, accuracy_val_gbdt/n_splits))
print("Accuracy of random forest model in trainset is {}, in validation set is {}".format(accuracy_train_rf/n_splits, accuracy_val_rf/n_splits))
```

    0.05898876404494388
    0.48884150675195454
    0.4878234952020937
    0.487050045539028
    0.5013738599416244
    0.4993072711061114
    0.48842879904533276
    0.500676123356637
    0.5216761952831759
    0.5112913827235928
    0.4774952147580528
    0.5225525181533575
    0.4894447280895803
    0.5098067794249237
    0.512196879262812
    0.5000413888060139
    0.477972700115165
    0.49904848521208117
    0.4893504310336947
    0.48935777774883515
    0.5208070265191829
    0.5004254235827161
    0.5098197238279855
    0.49286675739564073
    0.49579882572624867
    0.5031184753760831
    0.4891171507284799
    0.510617311942859
    0.48883153025536885
    0.49003279362202273
    0.5290150867977811
    0.5017886808509145
    0.49763607209025784
    0.4903072031716874
    0.5010493702724947
    0.4905904288867759
    0.5078164851635413
    0.5098241271719063
    0.47362524345950213
    0.5252538217050903
    0.5007611684622881
    0.48521159838876204
    0.5232285762881089
    0.47875799917498657
    0.49029023849992687
    0.509934866583859
    0.5010786372519085
    0.5007383670043614
    0.49032006053861693
    0.5189767132476576
    0.49687036933718676
    0.48323319159222605
    0.5088624535279672
    0.5099784380625294
    0.4818223138030813
    0.5070007518626325
    0.49306696506712594
    0.5000329522726222
    0.5068074610869527
    0.4932531908271561
    0.5073299440546873
    0.5099158306725355
    0.4897746075459546
    0.499759893149596
    0.5009091793996713
    0.5179370652940855
    0.48231210802474833
    0.5174738504068096
    0.48468982614077183
    0.4962367127749779
    0.5024424217719689
    0.5000000000000004
    0.5062094937386338
    0.5013704462619702
    0.49409049613194783
    0.5006177881720514
    0.49794073194498173
    0.5080125771047648
    0.4895780997460134
    0.5095363988632238
    0.4999999999999999
    0.5006427319509271
    0.49072925354495994
    0.5004606848865403
    0.5079312249409293
    0.4911779471353672
    0.5101059257167542
    0.49046342917903424
    0.5018020323971326
    0.49192779637661976
    0.5061882392487662
    0.5005992489780003
    0.5154035744257393
    0.4993641192300723
    0.49306094533977163
    0.4869578446625592
    0.5132400181531217
    0.4919381697179879
    0.5010559350078424
    0.49962550300480363
    0.06872370266479663
    0.5000000000000003
    0.49054893041553965
    0.38443914099034915
    0.6013775634201006
    0.5000000000000001
    0.4157122167615011
    0.5721286544545928
    0.499999999999999
    0.43696465480884567
    0.5559781421624198
    0.4999999999999995
    0.4999999999999996
    0.5000000000000008
    0.4999999999999995
    0.44965796501936406
    0.5013681825206456
    0.5
    0.49863555110959873
    0.501360735582091
    0.4999999999999994
    0.5445054975328011
    0.4593756998898267
    0.5373279853757152
    0.5000000000000008
    0.46550862342277666
    0.5320227097899966
    0.47014681373028955
    0.5156979143629228
    0.4708466113159515
    0.5139793126202149
    0.49797059133496985
    0.5282249746264592
    0.4732831880521937
    0.525361647228429
    0.4890625791853711
    0.48599008171249947
    0.5020391360931806
    0.5220719722468014
    0.46860691412611055
    0.4977330211991908
    0.511913556728267
    0.4884077071294202
    0.5307794067572245
    0.5000000000000003
    0.49151766075471326
    0.5083058367525615
    0.4918647481311268
    0.4999999999999997
    0.5079702804161758
    0.5000000000000001
    0.48187769797078156
    0.5173218393389012
    0.5
    0.4809563152163104
    0.518344973016699
    0.4744426786921496
    0.5097134670728741
    0.5148663705366193
    0.5
    0.48569991647581334
    0.5137666280936211
    0.4999999999999998
    0.4867368726105443
    0.5127870392221253
    0.4876638912375008
    0.5047406339762587
    0.4927727991278137
    0.4999999999999999
    0.5142796739778778
    0.4888982856421397
    0.5033944545072705
    0.5074086228241972
    0.5000000000000002
    0.5000000000000001
    0.5000000000000001
    0.4794409386795128
    0.519649888021749
    0.49999999999999956
    0.4877147357714764
    0.5119906476785202
    0.5000000000000003
    0.5000000000000002
    0.4934660824485406
    0.4974203340140379
    0.49999999999999994
    0.49353718285645876
    0.5063469590866415
    0.5000000000000006
    0.4999999999999992
    0.5089180944188653
    0.4999999999999994
    0.4884036545564133
    0.5050489201594865
    0.49805529985464886
    0.4970884951527445
    0.5111120154655822
    0.49209001155832527
    0.5000000000000006
    0.5012855142221899
    0.09817671809256662
    0.39985558764718904
    0.504978591363048
    0.5050480284742755
    0.5353281890709555
    0.5226980971140044
    0.49824352395508864
    0.4769756660265136
    0.5251784741298421
    0.4436827874394395
    0.5
    0.5
    0.5038049010575838
    0.5421443680148501
    0.45863452958897105
    0.5225474520515531
    0.49522260400828516
    0.5249072391643903
    0.5118809857381057
    0.4930360008141831
    0.4916637639313835
    0.5044897457841889
    0.4799441414768499
    0.5329501722267791
    0.46407356768097263
    0.5146544869469434
    0.47590064914668445
    0.4961868689780201
    0.5168776069085783
    0.5144259515856151
    0.4751339818318475
    0.5224014754331658
    0.4890437195465516
    0.5040678896424495
    0.5135518259737232
    0.48149546353505346
    0.4908812488940595
    0.5282631488459606
    0.4828017770901045
    0.49132107315441675
    0.5128125655771498
    0.49575618485796086
    0.499842953543718
    0.48876747178660923
    0.5
    0.5088621299898511
    0.5031116186332927
    0.4992673248660452
    0.5035553327671622
    0.5068013824700799
    0.4899881305928044
    0.48941864689494946
    0.5140686697318679
    0.49050612572144814
    0.5069198007248941
    0.5024888673317645
    0.5015307179436224
    0.4859009760853673
    0.4999999999999989
    0.5073715951425471
    0.49298606059675953
    0.4999999999999996
    0.5198061039961477
    0.49235604155922524
    0.49309091745898315
    0.5063203638296094
    0.49999999999999906
    0.4908208740473315
    0.49980229886665034
    0.5174922203414029
    0.4844730171822168
    0.5146780329086972
    0.49400659921854956
    0.49784145865406715
    0.5019414973601972
    0.49254572177357125
    0.5035205587757486
    0.5070135820526073
    0.4975818454903602
    0.49497008780050367
    0.504101404511287
    0.49304661389275056
    0.5000000000000009
    0.49999999999999883
    0.5029350079444439
    0.5039398886058526
    0.5029468426428907
    0.4992572470129248
    0.49112003937772586
    0.5117029412203599
    0.49551528415218155
    0.5008839145929987
    0.4920719714225092
    0.5109145137267138
    0.49903962567399895
    0.4960951670288983
    0.4976990877379871
    0.5067788500161927
    0.49441824033607645
    0.49762659491566413
    0.06732117812061711
    0.49033521303258215
    0.5094524354611097
    0.48997047179761904
    0.49993105423057443
    0.49016320538139946
    0.5096439268347508
    0.5099021849347
    0.5000000000000009
    0.490319336688571
    0.5000067742644609
    0.49053250266569526
    0.4991369159015663
    0.5101056304192513
    0.4909342305178023
    0.5092919149314439
    0.5007184611909805
    0.48937591309313805
    0.5096213426038393
    0.4905894863437257
    0.4914630385561255
    0.5171434651969763
    0.4921425693852941
    0.5073900453524048
    0.491562439523662
    0.5094256654394765
    0.4825671586859247
    0.5077494628750188
    0.5007775265816738
    0.5092134250906186
    0.4995024633400163
    0.49102123740640935
    0.5087913405968024
    0.4922195131672811
    0.5076329268550651
    0.49214775581618914
    0.5000711299908009
    0.5073459135586259
    0.49278848885381166
    0.4986273704260908
    0.5007761537441202
    0.5076666514075591
    0.5015560130240284
    0.49151571760856805
    0.5067461942878974
    0.4925552661689164
    0.49920118615613596
    0.49999999999999883
    0.5015399683475262
    0.5008000036329465
    0.5058861357929153
    0.48516065739572634
    0.5007849443756329
    0.5057716322159683
    0.5162227696783707
    0.492335511565872
    0.4936853435184544
    0.4991841035755159
    0.4987282471257396
    0.5012314610420415
    0.49257043120609756
    0.5142202351651219
    0.49216781895625683
    0.5007945138815063
    0.4992067466201368
    0.4939891398268531
    0.513481999875412
    0.5018375801110622
    0.49330089768007385
    0.5056191773033012
    0.501151826182324
    0.48523985093253186
    0.5007877257369403
    0.5045452859372423
    0.5071168930015832
    0.4938161865732361
    0.500815508724583
    0.49313561900187036
    0.5072652117582456
    0.5049796917341947
    0.49999999999999944
    0.488023524376763
    0.5070639918261904
    0.5067010685561877
    0.4934152511362562
    0.504627396639426
    0.4955868258074952
    0.5042091969554866
    0.5009062694150035
    0.49368692610848136
    0.5061565451671522
    0.5002570363203629
    0.48795777011611413
    0.5047292875435341
    0.5083142620366686
    0.4989778210450163
    0.5057887953369433
    0.4874503303573888
    0.4950989937325474
    0.5048280745441548
    0.08134642356241234
    0.5000000000000003
    0.4921426691234537
    0.4999999999999993
    0.49999999999999983
    0.49999999999999994
    0.49999999999999994
    0.5
    0.5
    0.5
    0.5077117681552615
    0.49242929695135956
    0.5000000000000008
    0.5074339226486558
    0.4926987725737229
    0.49999999999999994
    0.49999999999999994
    0.49999999999999994
    0.5071724301977143
    0.4999999999999997
    0.49295264481499007
    0.5069258371610749
    0.4931922793295726
    0.4999999999999995
    0.5066928593179595
    0.49341888488112945
    0.5064723579037474
    0.4936335352266179
    0.5062633195985577
    0.49999999999999967
    0.5
    0.49383718743946237
    0.506064839728437
    0.49403069732993277
    0.5000000000000008
    0.5058761080932054
    0.5000000000000001
    0.49421483248543824
    0.5000000000000001
    0.4999999999999997
    0.5000000000000001
    0.5056963969539493
    0.49999999999999994
    0.49439028334851387
    0.5055250508058003
    0.49455767266854783
    0.5000000000000006
    0.5053614776339543
    0.4947175635990922
    0.5000000000000001
    0.5000000000000001
    0.5000000000000001
    0.505205141407924
    0.49999999999999956
    0.4999999999999998
    0.49487046666185946
    0.5050555556142143
    0.49999999999999956
    0.4950168457581984
    0.5000000000000002
    0.5000000000000002
    0.5000000000000001
    0.5000000000000001
    0.5000000000000001
    0.5
    0.5
    0.5
    0.5
    0.5049122776636339
    0.4951571233766526
    0.5000000000000003
    0.5047749040383157
    0.4952916851191928
    0.5046430660668265
    0.5000000000000008
    0.495420883647913
    0.5000000000000002
    0.5000000000000002
    0.5000000000000001
    0.5000000000000001
    0.5000000000000001
    0.5045164262345536
    0.5
    0.5
    0.49554504213687856
    0.4999999999999989
    0.5000000000000011
    0.4999999999999989
    0.5000000000000011
    0.5043946749519376
    0.4956644572999958
    0.5042775277156576
    0.4957794020544
    0.500000000000001
    0.49999999999999883
    0.500000000000001
    0.5041647226081577
    0.4958901278695096
    0.49999999999999967
    0.5000000000000013
    Accuracy of cart model in trainset is 0.9295767212443071, in validation set is 0.7946331052664617
    Accuracy of adaboost model in trainset is 0.9295767212443071, in validation set is 0.7991274872889336
    Accuracy of gbdt model in trainset is 0.9635249066296862, in validation set is 0.8148138848785387
    Accuracy of random forest model in trainset is 0.9082500748538381, in validation set is 0.8294520118008913
    

观察上述结果，我们可以发现AdaBoost和GBDT模型主要关注降低错误率（即降低偏差），因此他们能基于分类性能相当弱的学习器构建出分类性能很强的分类器；Bagging主要关注降低方差，因此Random Forest得到的结果在验证集上表现较好，不容易过拟合。

到这里我们本次作业就结束了，有兴趣的同学可以用我们生成的模型去拟合kaggle测试集的数据，提交结果得到最终成绩。

当然这个模型还有很大的提升空间，首先是数据方面，采用更高级的方式补充缺失值，保留姓名字段等方式都可以得到更充足的数据信息，其次模型选择，弱学习器的选择，一些超参数的调节，这些都会对最终的模型产生影响，你也可以采用上述策略去提升kaggle竞赛的最终成绩。


```python

```
