[toc]



# 支持向量机SVM

支持向量机（Support Vector Machine, SVM）通过核函数将低维数据映射成高维，然后确定最优的划分超平面。能够处理在低维空间中难以分割的情况。

## 思路

假设数据集为$D=\{(x_i,y_i)\}$，其中样本量为$n$，$x_i$是$d$维特征，$y_i$仅为$1$或$-1$，表示正类与负类。我们的目标是找到一个超平面（假设$d=2$，那么$x_i$就是平面上的点，此时就是找一条直线）将正负两类样本分开，而且这两类样本之间的“距离”最大。

为了描述两类的“距离”，首先定义间隔。对于一个确定的超平面$w^Tx+b=0$，它关于样本点$(x_i,y_i)$的函数间隔就定义为
$$
\hat \gamma_i=y_i \cdot (w^T x_i+b)
$$
假设超平面的两侧分别是预测的正类与负类，那么预测正确的样本与预测错误的样本，其函数间隔的正负号不同，且函数间隔的数值越大，预测的确信度（也即分类正确或错误的程度）也就越大。我们不妨将$w^T x_i+b>0$一侧的样本点预测为正类，另一侧为负类。此时函数间隔为正的，代表预测正确，为负的代表预测错误。而对于预测正确的样本点，函数间隔的值越大，就说明样本点距离超平面越远。

由于超平面的表达式不唯一，为了消除系数对间隔大小的影响，我们定义几何间隔为
$$
r_i=\frac{\hat \gamma_i}{\|w\|_2}=\frac{y_i \cdot (w^T x_i+b)}{\|w\|_2}
$$
此时，每个样本的几何间隔，就是该样本到目标超平面的垂直距离。现在，我们要找到使得几何间隔尽量大的超平面。

在超平面两侧，都会有距离超平面最近的点，称为支持向量。目标超平面将由这些支持向量决定（而不是全部的样本）。使得训练样本到超平面的最小距离（也即支持向量到目标超平面的距离）最大化，这样的超平面就是目标超平面。用数学语言描述，就是求$w,b$达到
$$
\max\limits_{w,b} \min\limits_{x_i} \frac{y_i \cdot (w^T x_i+b)}{\|w\|_2}
$$
不妨令支持向量满足$y_i(w^T x_i+b)=1$（通过调整$w,b$总是可以做到），该问题其实等价于求
$$
\min\limits_{w,b} \frac{1}{2}\|w\|^2_2,\ \text{ s.t. } \ y_i(w^T x_i+b)\geqslant1
$$
这是一个带有线性约束的二次规划问题，可以用牛顿法等数值方法解决。

如果在当前的$d$维空间中找不到合适的超平面，还可以尝试将每个样本$x_i$映射成更高维度的向量，在新的空间中找。当然实际计算中并不会去真的映射，升维是在算法中，通过修改核函数体现出来的。

## 算法

### 原始版本

线性规划问题往往转换成对偶问题来解决。首先用拉格朗日乘子法将原问题变为无约束的最优化问题：令拉格朗日函数为$L(w,b,\alpha)=\frac{1}{2}\|w\|^2_2-\sum\limits_i \alpha_i(y_i(w^T x_i+b)-1)$，那么问题转化为
$$
\min\limits_{w,b} \max\limits_{\alpha_i\geqslant0} L(w,b,\alpha)
$$
它的对偶问题就是求
$$
\max\limits_{\alpha_i\geqslant0} \min\limits_{w,b} L(w,b,\alpha)
$$
通常对偶问题的极值，会超越原问题的极值，为了确保对偶问题和原问题的解相同，需要满足斯莱特条件（此时对偶问题与原问题的解相同，称为强对偶性）。由于我们的原问题是凸二次规划，已经满足了斯莱特条件。

接下来，还需要满足KKT条件（在强对偶性下，它是最优解满足的必要条件），才能求出最优解。在这个问题里，KKT条件中的一部分已经天然满足，还需要补充的是
$$
\begin{align}
\frac{\partial L(w,b,\alpha)}{\partial w} & =w-\sum\limits_i\alpha_iy_ix_i=0 \\
\frac{\partial L(w,b,\alpha)}{\partial b} & =-\sum\limits_i\alpha_iy_i=0 
\end{align}
$$
将$w=\sum\limits_i\alpha_iy_ix_i$代入拉格朗日函数，则最优化问题等价于
$$
\begin{align}
\max\limits_{\alpha_i\geqslant0} &&& \sum_{i}^n \alpha_i-\frac12 \sum_i\sum_j\alpha_i\alpha_jy_iy_j(x_i^Tx_j), \\
\text{s.t.} &&& \sum_i^n\alpha_iy_i=0,\ \alpha_i\geqslant0.
\end{align}
$$
现在，求解目标转变成了$\alpha_1,...,\alpha_n$。为了解决这个转化后的问题，微软研究院的John C. Platt提出了SMO算法。其基本思路是，初始化全部$\alpha_i$后，选择两个$\alpha_j,\alpha_k$，固定其他变量，关于这两个变量求解最优化问题，获得更新后的$\alpha_j,\alpha_k$。这样反复更新$\alpha_i$们，直到结果收敛。如此便可以求出$\alpha_1,...,\alpha_n$。

这之后，利用$w=\sum\limits_i\alpha_iy_ix_i$可以求出$w$；由支持向量满足$y_i(w^T x_i+b)=1$可以解出$b$。这就得到了原问题的解。

原问题的时间复杂度为$O(d^3)$，$d$是特征维数，适用于特征维数较小的情况；对偶问题时间复杂度为$O(n^3)，$适用于样本数较少的情况。

### 改进-核函数

在一些局面下，原本的样本并不是线性可分的，考虑将它们映射到高维空间来变成线性可分。这称为使用核方法。

在思路中，需要把样本$x$用升维后的$\phi(x)$代替。但我们注意到，计算对偶问题的过程中，涉及$x$的部分实际只有计算内积$x_i^Tx_j$，所以只需要考虑把$x_i^Tx_j$改换形式成$\varphi(x_i,x_j)=\phi(x_i)^T\phi(x_j)$即可。

通常用的有如下核函数

| 类型    | 函数$\phi$                                                   | 核函数$\varphi$                            | 说明                                                        |
| ------- | ------------------------------------------------------------ | ------------------------------------------ | ----------------------------------------------------------- |
| 线性    | $\phi(x)=x$                                                  | $x_1^Tx_2$                                 | 其实就没变                                                  |
| 多项式  | $\phi(x)=(p_1(x),...,p_m(x))$，其中每个$p_m(x)$是向量$x$的多项式 | $(\gamma x_1^Tx_2+r)^d$                    | 正整数$d$代表多项式最高次数                                 |
| 高斯    | $\frac1{\sqrt{2\pi}\sigma}\exp(-\frac{(x-\mu)^2}{2\sigma^2})$ | $\exp(-\frac{\|x_1-x_2\|^2_2}{2\sigma^2})$ | $\sigma>0$                                                  |
| sigmoid | $\frac 1 {1+e^{-x}}$                                         | $\tanh(\beta x_1^T x_2+\theta)$            | $\beta>0,\theta<0$，适合数据分布不明确或呈现 S 形模式的情况 |

> 来自Sklearn文档 [Plot classification boundaries with different SVM Kernels](https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html#sphx-glr-auto-examples-svm-plot-svm-kernels-py)

将向量映射到高维空间，这个“映射”本身肯定带有参数，所以映射后的核函数中也包含参数，描述了不同的映射。这个参数是模型参数的一部分，会在求解模型的时候被确定下来（我猜的，待进一步查证）。

### 改进-软间隔

在实际中，样本未必真的线性可分，即使用核方法强行分开了，也可能是过拟合。我们需要在一定程度上容忍“错判”的样本，来保证结果的可靠性。所以，我们允许一些数据跨越分类超平面，去往错误的类别，只是对它们施加一定的惩罚。这就是“软间隔”（soft margin）的思想。

在计算时，优化目标改变为
$$
\min\limits_{w,b,\xi} &&& \frac{1}{2}\|w\|^2_2+C\sum_i l(y_i(w^T x_i+b)-1), \\
$$
其中的$ l(z)=I_{\{z<0\}}$，也即$y_i(w^T x_i+b)<1$将计入惩罚。

由于$I_{\{z<0\}}$是非凸、非连续的，所以我们希望用其他函数来作为$l(z)$。最常用的软间隔支持向量机采用了hinge损失：$l(z)=max(0,1-z)$。此时令$\xi_i=l(y_i(w^T x_i+b)-1)$，最初的问题就相当于
$$
\begin{align}
\min\limits_{w,b,\xi} &&& \frac{1}{2}\|w\|^2_2+C\sum_i \xi_i, \\
\text{s.t.} &&& y_i(w^T x_i+b)\geqslant1-\xi_i,\ \xi_i\geqslant0,&i=1,2,...,n.
\end{align}
$$
之后的推导方法完全相同，最终得到的对偶问题变为
$$
\begin{align}
\max\limits_{\alpha_i\geqslant0} &&& \sum_{i}^n \alpha_i-\frac12 \sum_i\sum_j\alpha_i\alpha_jy_iy_j(x_i^Tx_j), \\
\text{s.t.} &&& \sum_i^n\alpha_iy_i=0,\ 0\leqslant\alpha_i\leqslant C,\ i=1,2,...,n.
\end{align}
$$
同样可使用SMO算法求解。



# 支持向量回归

支持向量回归（Support Vector Regression, SVR）将支持向量机的思路用于回归。对数据集$D=\{(x_i,y_i)\}$，希望得到一个线性模型$ \hat y=f(x)=w^Tx+b$来描述。（实际上通常会使用核方法，得到的是非线性函数$f(x)=w^T\phi(x)+b$。）

在进行预测时，允许预测值与真实值之间有$\varepsilon$的误差，也即，只有$f(x)$与$y$之间的差的绝对值大于$\varepsilon$时才计算损失。SVR的目的便是求
$$
\min\limits_{w,b} \frac12 \|w\|^2+C\sum_{i=1}^n l(y-f(x))
$$
其中
$$
l(z)=\begin{cases}
0 & |z|\leqslant\varepsilon \\
|z|-\varepsilon & |z|>\varepsilon
\end{cases}
$$
是损失函数。$C$是正则化参数。

我们注意到，这个求解目标与软间隔支持向量机的形式完全类似，同样是转化成对偶问题后求解。推导可以参考https://zhuanlan.zhihu.com/p/76609851

## 代码

官方文档：https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR

```Python
#签名
class sklearn.svm.SVR(*, kernel='rbf', 
                      degree=3, 
                      gamma='scale', 
                      coef0=0.0, 
                      tol=0.001, 
                      C=1.0, 
                      epsilon=0.1, 
                      shrinking=True, 
                      cache_size=200, 
                      verbose=False, 
                      max_iter=- 1)
#范例
>>> from sklearn.svm import SVR
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.preprocessing import StandardScaler
>>> import numpy as np
>>> n_samples, n_features = 10, 5
>>> rng = np.random.RandomState(0)
>>> y = rng.randn(n_samples)
>>> X = rng.randn(n_samples, n_features)
>>> regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
>>> regr.fit(X, y)
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('svr', SVR(epsilon=0.2))])
```



超参数：

​	kernel：核函数。默认使用rbf（径向基函数），还可以是‘linear’, ‘poly’, ‘sigmoid’, ‘precomputed’或者手动给出的可调用对象（callable）。

​	degree：如果使用多项式核函数，该参数用于指定多项式次数。

​	gamma：如果使用径向基函数、多项式函数、Sigmoid函数作为核函数，需要该参数。

​	C：惩罚函数的正则化参数。必须是正值。默认1。

​	epsilon：预测的允许误差限。预测误差小于这个值的样本点，不会计入损失。

​	max_iter：求解时的迭代次数限制。默认为-1也即无限制。

​	tol：迭代的停止精度。默认1e-3。

属性：

​	support_vectors_：支持向量。以矩阵形式给出，每行是一个支持向量。

方法：

​	fit(*X*, *y*, *sample_weight=None*)：训练模型。

​	get_params()：给出系数。

​	predict(*X*)：预测。

​	score(X, y, sample_weight=None)：计算判定系数。



# 单分类支持向量机One-Class SVM

One-Class SVM (OCSVM) 是一种专门用于异常检测或新颖性检测的无监督机器学习算法。与传统的 SVM 用于分类不同，OCSVM 的目标是学习一个能够将绝大多数正常数据点包围起来的决策边界，而位于这个边界之外的数据点则被视为异常或离群点。

在单分类SVM中，会假设数据中正常样本很多且有着共同的分布（属于同一集群），如果能像SVM一样找到一个超平面来把这些正常点和“其他”分布的点隔离开，使得正常点都在超平面的正向一侧就好了。

但SVM的目标是最大化两个类别间的距离，而单分类SVM 只有一个类别，于是我们考虑用原点（代表“空”分布）作为OCSVM的负类别。这样一来，就可以沿用SVM的优化目标，希望找到一个超平面$w^Tx-\rho=0$满足$w^Tx_i-\rho>0$（样本$x_i$在其正侧），同时距离原点尽量远。一个平面$w^Tx+b=0$到原点的距离是$\frac{|b|}{\|w\|}$（这里需要回顾解析几何的知识），因此我们需要最大化$\frac{|\rho|}{\|w\|}$。又因为原点在超平面负侧，应该有$w^T\cdot0-\rho< 0$也即$\rho>0$。

目标$\min \frac\rho{\|w\|}$等价于$\min \frac12\|w\|^2-\rho$，后者具有更容易求导的形式。假设共$N$个样本，此时的优化目标为
$$
\min \frac12\|w\|^2-\rho \\
\text{ s.t. }\ 
w^T x_i >\rho \ \ (i=1,2,...,N)
$$
在这个目标中，$\rho$描述了超平面与原点的距离，$w$描述了原点和数据点之间的“边界形状”。

找到一个完美的超平面是困难的，像SVM一样，OCSVM引入“允许错误”的思想，为第$ i $个样本点$ x_i $引入松弛变量$\xi_i$来衡量允许其违反约束条件 $w^Tx_i≥ρ $的程度，$\xi_i$的范围为$[0,+\infty]$。这样就要在优化目标中加上新的损失部分$\frac1N\sum\xi_i$，表示对“越界”的惩罚。为了控制这种惩罚的程度，再加入超参数$\nu$变成$\frac1{\nu N}\sum\xi_i$，$\nu$越小表示对错误越敏感、越不允许错误出现。同时原本要满足的条件$w^T x_i >\rho$也放松为$w^T x_i >\rho-\xi_i$。

最终，得到的优化目标为
$$
\min \frac12\|w\|^2 + \frac1{\nu N}\sum_{i=1}^N\xi_i - \rho \\
\text{ s.t. }\ 
w^T x_i > \rho - \xi_i \ \ (i=1,2,...,N)
$$
在这个目标中，$w,\rho,\xi_i$是优化对象，而$\nu$是超参数。

分析一下$\nu$的物理意义。它是一个比例，因此在$(0,1]$之间。当$\nu=1$时所有样本越界的惩罚都正常计算，而$\nu$很小意味着微小的惩罚被严重放大，所以$\nu$可以看作“允许犯错的比例”。如果对数据集中异常点的预期比例有一个大致的了解（例如，已知异常点大约占$ 1\%$），可以尝试将$\nu$设置为接近这个值。

# SVDD

SVDD（Support Vector Data Description）是另一种单分类方法。相比于OCSVM，SVDD 希望找到一个超球体而不是超平面来包裹所有正常样本。超球可以用中心$ a $和半径$ R>0$来描述。如果能最小化$ R^2 $，并保证球体包含所有样本$ x_i$，就同样找到了合适的边界。该方法与 OCSVM 求解的超平面相似，最优化目标如下：
$$
\min R^2 + C\sum_{i=1}^N\xi_i \\
\text{ s.t. }\ 
\| x_i - a \| \le R^2 + \xi_i \ \ (i=1,2,...,N)
$$
其中$\xi_i$和OCSVM一样是松弛变量，$C$用于控制误差的容忍度。