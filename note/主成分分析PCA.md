[toc]



# 概述

主成分分析 (principal components analysis) 是一种用来降维的方法。说是降维，实际上是对原数据做了一个变换，变换后得到的新数据会有一些独特的性质。

降维的目的，是针对随机向量/每组样本，用低维的统计量$(y_1,y_2,\cdots,y_m)$代替高维的原数据$(x_1,x_2,\cdots,x_p)$。对随机变量$x=(x_1,x_2,\cdots,x_p)'$，考虑让每个$y_i$都与$x$有线性关系，也即$y_i=a_i'x$，其中每个$a'_i$是列向量。这里需要注意的是，概率论中的随机向量$x$是列向量；而统计中的数据集$X$，其每一行是一个样本，样本可看作是随机向量的采样，这样一来，$x$又宜表示成行向量。要注意甄别记号$x$的含义与其形状。

考虑这样的条件：首先，为了使得$y_i$保留$x$的信息，应该让$y_i$的方差（在$x$的一切线性组合中）最大化；其次，为了使得主成分之间有可比性，需要限制$a_i$为单位向量；最后，为了避免$y_i$之间信息产生重复，应该使它们两两之间（实际上只需每一个与之前的相比）协方差为$0$。

# 总体的主成分

假设$p$维随机向量$x$的均值为$0$，协方差矩阵$D(x)=\Sigma$。根据上述想法，$a_1$就是在$a_1'a_1=1$条件下使得$D(y_1)=D(a'_1x)=a_1'\Sigma a_1$最大，使用Largrange乘子法，会发现此时$a_1$恰应该是$\Sigma$的单位特征向量，而要保证$a_1'\Sigma a_1=\lambda a_1'a_1=\lambda_1$最大，$a_1$应该是最大特征值对应的特征向量，$\lambda_1$是最大特征值。再结合实对称阵一定可对角化，因而可以得到两两正交的单位特征向量，可以发现，主成分$y_i=a_i'x$们需要的的$a_i$，恰恰是随机变量$x$的协方差矩阵$\Sigma$的全部单位正交特征向量，而且要按照对应特征值的降序排列。

也即，假设$\Sigma$的特征值为$\lambda_1 \geqslant \lambda_2 \geqslant \cdots \geqslant \lambda_p $（可以证明这些特征值都是非负的），那么它们对应的单位正交特征向量$t_1,t_2,\cdots,t_p$就是满足我们需求的$a_1,a_2,\cdots,a_p$（实践中常常只取前几个）。由此得到的$y_i=t_i'x$就称为第$i$主成分。写成矩阵形式，就是
$$
y = \begin{pmatrix}
y_1 \\ y_2 \\ \vdots \\y_p
\end{pmatrix} =
\begin{pmatrix}
t_{11} & t_{21} & \cdots & t_{p1} \\
t_{12} & t_{22} & \cdots & t_{p2} \\
\vdots & \vdots & \ddots & \vdots \\
t_{1p} & t_{2p} & \cdots & t_{pp} \\
\end{pmatrix}
\begin{pmatrix}
x_1 \\ x_2 \\ \vdots \\x_p
\end{pmatrix} =
\begin{pmatrix}
t_1' \\ t_2' \\ \vdots \\ t_p' \\
\end{pmatrix} x = T'x, \\
\Sigma=T\text{diag}(\lambda_1,...,\lambda_p)T'
$$
其中$T$是以$t_1,t_2,\cdots,t_p$为列的正交矩阵，$t_{ij}$理解为第$j$主成分关于第$i$分量的系数，它也是第$j$特征向量的第$i$分量。

改写这个式子可以得到其他形式的结果：
$$
x = Ty=(t_1,t_2,...t_p)y \\
x_i = t_{i1}y_1+t_{i2}y_2+\cdots+t_{ip}y_p
$$
对于一般的$x$，考虑到实际数据的尺度差异会比较大，而分量之间的数值大小差异过大（因而方差也差异过大）时，主成分很大程度上受数值大的分量影响，数值小的分量会被忽略。为了修正，我们先将每个分量（特征）标准化。最常用的标准化方式是正态化，也即$x_i^*=\displaystyle\frac{x_i-\mu_i}{\sqrt{\sigma_{ii}}}$，这时$x^*=(x_1^*,x_2^*,\cdots,x_p^*)'$的协方差矩阵正是原本的$x$的相关矩阵$R=\operatorname{cor}(x)$，用$R$替换$\Sigma$即可计算出改进的主成分$y^*=(y_1^*,y_2^*,\cdots,y_p^*)'$，相应的特征值与特征向量记为$\lambda_1^*,\lambda_2^*,\cdots,\lambda_p^* $与$t_1^*,t_2^*,\cdots,t_p^* $。

# 总体主成分的性质

为了衡量主成分对原本数据的描述效率，需要研究主成分的性质，包括以下内容：

1. （主成分贡献率）各个主成分$y_1,y_2,\cdots,y_p$互不相关（因为是单位正交特征向量），方差$D(y_i)=\lambda_i$，而方差之和恰等于$x$的协方差矩阵$\Sigma$的对角元之和（由$\tr D(z)=\tr\Sigma$得出）。由此可见，协方差矩阵的迹是变换前后的不变量。所以我们用$  \lambda_i/\displaystyle\sum_{j=1}^p\lambda_j$来衡量第$i$主成分的解释效率，称之为贡献率。

1. （主成分与原始变量的相关系数）原始数据$x$的分量$x_i$与主成分$y_j$之间的相关系数可以由关系$x_i=\displaystyle\sum_{j=1}^pt_{ij}y_j$得出，计算可得
   $$
   \operatorname{Cov}(x_i,y_j)=\operatorname{Cov}(t_{ij}y_j,y_j)=t_{ij}\lambda_j \\
   \rho(x_i,y_j)=\frac{\operatorname{Cov}(x_i,y_j)}{\sqrt{V(x_i)}\sqrt{V(y_j)}}
   	=\frac{\sqrt{\lambda_j}}{\sqrt{\sigma_{ii}}}t_{ij}
   $$
   这个相关系数称为因子载荷量。

1. （主成分对原始变量的贡献率）由于$x_i$是$y_1,y_2,\cdots,y_p$的线性组合，所以复相关系数$\rho_{i\cdot1,2,\cdots,p }=1$，复相关系数的平方又是每个分量的相关性系数的平方和，所以$\rho_{i\cdot1,2,\cdots,p }^2=\displaystyle\sum_{j=1}^p\rho^2(x_i,y_j)=1$。也可以由$\sigma_{ii}=\displaystyle\sum_{j=1}^pt_{ij}^2\lambda_j$直接推出$\displaystyle\sum_{j=1}^p\rho^2(x_i,y_j)=\displaystyle\sum_{j=1}^p\frac{\lambda_j}{\sigma_{ii}}t_{ij}^2=1$。这启示我们，如果要考虑前$m$个主成分$y_1,y_2,\cdots,y_m$对$x_i$的反映程度，可以用$\rho_{i\cdot1,2,\cdots,m }^2=\displaystyle\sum_{j=1}^m\rho^2(x_i,y_j)=\displaystyle\sum_{j=1}^m\frac{\lambda_jt_{ij}^2}{\sigma_{ii}}$来衡量，这称为前$m$个主成分对$x_i$的贡献率。

1. （主成分的决定因素）主成分$y_j$与原始变量$x_i$之间有关系$y_j=\displaystyle\sum_{i=1}^pt_{ij}x_i$，称$t_{ij}$为第$j$主成分在第$i$原始变量上的载荷。还可证明$\sigma_{ii}=\displaystyle\sum_{j=1}^pt_{ij}^2\lambda_j$，这说明$x_i$的方差$\sigma_{ii}$是$y_j$的方差$\lambda_j$的加权平均（因$T$是正交矩阵），于是较大的$y_i$与方差较大的$x_i$相关性更强，较小的$y_i$与方差较小的$x_i$相关性更强。当这种效应十分明显时，我们会注意到主成分几乎就是原始数据按方差大小进行的重新排列。

1. （主成分揭露共线性）当$\lambda_j$非常小（接近$0$）时，可以认为$y_j$是常数，这又意味着$\displaystyle\sum_{i=1}^pt_{ij}x_i$是常数，也即$x_i$之间存在共线性。这可以看作一种检查共线性、筛选变量的方法。

1. （相关系数矩阵下的主成分）考虑使用标准化修正后的主成分，上述性质会发生改变，主要体现在新的$x_i$的方差$\sigma_{ii}^*$会变成$1$，上述公式中的$\lambda_j$与$t_j$也相应地用$\lambda_j^*$与$t_j^*$代替。这样做完全等效于使用相关系数矩阵来代替协方差矩阵计算主成分。由于方差的差异被消除了，上述的3.4.点不再体现出来。

# 样本主成分

以上所描述的，均是某个确定总体下的主成分。而实际上我们是在用样本估计总体，只能求出样本的主成分。假设数据矩阵是
$$
X=
\begin{pmatrix}
x_1' \\ x_2' \\ \cdots \\ x_n' 
\end{pmatrix}=
\begin{pmatrix}
x_{11} & x_{12} &\cdots &x_{1p} \\
x_{21} & x_{22} &\cdots &x_{2p} \\
\vdots & \vdots &\ddots &\vdots \\
x_{n1} & x_{n2} &\cdots &x_{np}
\end{pmatrix}
$$
那样本协方差矩阵为$S=\displaystyle\frac{1}{n-1}\sum_{i=1}^n(x_i-\bar x)(x_i-\bar x)'=(s_{ij})$，样本相关矩阵为$\hat R=(r_{ij})$，其中$r_{ij}=\displaystyle\frac{s_{ij}}{\sqrt{s_{ii}}\sqrt{s_{jj}}}$是样本间的相关性系数，$\bar x=\displaystyle\frac1n\sum_{i=1}^nx_i$是样本均值。用$S$、$\hat R$作为$\Sigma$、$R$的估计，即可算出样本的主成分。

此时，对一个样本$x_k'$，它的得分向量为
$$
y_k'=(y_{1k},\cdots,y_{pk})=x_k'T = (x_{1k},\cdots,x_{pk}) 
\begin{pmatrix}
t_{11} & t_{12} & \cdots & t_{1p} \\
t_{21} & t_{22} & \cdots & t_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
t_{p1} & t_{p2} & \cdots & t_{pp} \\
\end{pmatrix}
$$
全部样本的全部主成分得分可以写为
$$
Z =
\begin{pmatrix}
y_1' \\ y_2' \\ \vdots \\y_n'
\end{pmatrix} =
\begin{pmatrix}
x_{11} & x_{12} & \cdots & x_{1p} \\
x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{p1} & x_{p2} & \cdots & x_{pp} \\
\end{pmatrix} \begin{pmatrix}
t_{11} & t_{12} & \cdots & t_{1p} \\
t_{21} & t_{22} & \cdots & t_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
t_{p1} & t_{p2} & \cdots & t_{pp} \\
\end{pmatrix}
$$




# 应用

1、样本降维

这是最直观的使用方式，暂不详述。

2、特征分类

（用相关性矩阵$R$算出的）第$k$主成分$Z_k$与第$i$特征$X_i$的相关性系数为$\sqrt{\lambda_k}a_{ik}$，其中$a_{ik}$是第$k$特征向量的第$i$分量，记其为$\rho_{ik}$. 如果某两个特征$X_i,X_j$对每个主成分的相关性系数都差不多，也即$\rho_{ik}\approx\rho_{jk}(k=1,2,...,m)$，那么可想而知，这两个特征是高度线性相关的。所以，可以用$Q_i=(\rho_{i1},\rho_{i2},...,\rho_{im})$作为第$i$特征$X_i$的一种代表，用点$Q_i$（的散点图等）来对特征进行分类。

3、特征重构

（在对含义加以解释后）使用主成分作为新的特征，可以进行排序和回归等。进行回归的好处在于，主成分消除了共线性。

4、正态性检验

多元正态分布的检验，通常要转化成一元正态分布检验来处理。当样本的各个分量之间存在相关性时，无法直接对各个分量做一元正态分布检验。转化成主成分则可以。