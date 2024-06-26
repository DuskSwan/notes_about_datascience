本文在ChatGPT的帮助下完成。

# 参考

核心思路：https://zhuanlan.zhihu.com/p/116908556

给出了存储图片的例子：https://blog.csdn.net/weixin_60737527/article/details/124707538

# 简介

Hopfield网络是由约翰·霍普菲尔德（John Hopfield）于1982年提出的一种受神经科学启发的递归神经网络，提出背景是对人脑的记忆和计算能力的研究。霍普菲尔德希望通过一种数学模型来模拟人脑如何存储和检索信息。他的目标是构建一个能够模拟联想记忆的神经网络系统，即能够从部分或噪声数据中恢复完整记忆的能力。

Hopfield网络的结构可以看作一张无向图，节点称为其神经元，每条边上则有权重。神经元通过对称的连接权重相互连接。从某一状态开始，根据权重不断更新神经元的值，网络的状态也会随之改变。同时Hopfield网络引入了一个能量函数（听起来就像损失函数？），用于描述系统的状态。网络状态的演化过程是一个逐渐降低能量的过程，最终网络会收敛到一个局部能量最小值，即稳定状态。

Hopfield网络的主要应用场景是联想记忆（associative memory），也称为内容寻址存储（content-addressable memory）。我们希望构建一个有着特殊权重的网络，每当给定一个输入（不完全的记忆模式或者带有噪声的内容），将其作为网络初始状态，都可以通过不断更新神经元状态，使网络状态逐渐演化到一个稳定的记忆模式（局部最小能量状态），这便是与输入最接近的记忆模式。

此外，Hopfield网络还能应用于组合优化问题。其基本思想是将优化问题的目标函数转换为网络的能量函数。通过最小化能量函数，网络状态自然演化到最优解或接近最优解的位置。

Hopfield神经网络可分为离散型Hopfield神经网络（Discrete Hopfield Neural Networks，DHNN）和连续型Hopfield神经网络（Continuous Hopfield Neural Networks，CHNN）两种。 二者的神经元值类型、激活函数、更新策略稍有不同，前者主要用于联想记忆，后者主要用于优化计算。

# 算法

此处**只考虑离散的网络**，每个神经元的取值仅为1或-1。这时的能量函数定义为
$$
E=−\frac12∑_{i,j}w_{ij}s_is_j+∑_iθ_is_i
$$

其中，$s_i$是神经元$i$的状态，$w_{ij}$是神经元$i$与$j$之间的连接权重，$θ_i$是偏置。

考虑用这样的网络来来储存信息。例如要储存n个长度为m的序列，那么我们就需要一个有m个节点的神经元的离散型Hopfield神经网络DHNN来储存。比如一张黑白图像，假设图像的像素值为0（黑）和1（白），可以将它们转换为-1（黑）和1（白），然后拉平成向量来表示。现在，一个好的网络应该接收一个输入，并且将储存下来的信息（这五张图片中的某一个）作为输出。

对一个DHNN，假设我们以某种方式计算出了它的权重（以及其他需要的参数），这个网络就确定下来。接下来我们按照如下流程来使用该网络：

1. 对网络进行初始化，为每个神经元赋初始状态。这个初始状态可以看作是网络的输入。它的含义是一个残缺的储存对象（或曰不完全的记忆内容）。

2. 从网络中随机或按照顺序选取一个神经元。

3. 更新该神经元的状态，其他神经元的状态保持不变。

4. 求当前状态下网络的能量，判断网络是否达到稳定状态，若达到稳定状态或满足给定条件（如限定迭代次数）则结束；否则转到第2步继续运行。

在这个流程中，还有两件事我们没有明确，那就是如何计算权重，以及如何更新网络状态。接下来我们具体描述这两步。

### 权重的确定——Hebbian学习规则

Hopfield神经网络常见的学习方法有Hebbian法、Storkey法和伪逆法。这里主要介绍Hebbian法。

Hebbian学习规则是一种基于神经活动同步性的学习机制，由加拿大心理学家唐纳德·赫布（Donald  Hebb）于1949年提出。它的基本思想是“用则强，不用则退”（Cells that fire together, wire  together），即如果两个神经元经常同时活动，它们之间的连接强度会增强。

具体来说，Hebbian学习规则描述了神经元之间突触权重如何根据它们的活动相关性进行调整。假设有$n$个模式（向量）要储存，分别记作$x^{(k)}=(x_1^{(k)}, \dots, x_m^{(k)}),1\le k\le n$。那么权重$w_{ij}$的计算公式为

$$
w_{ij} = \begin{cases}
\sum_{k=1}^n x_i^{(k)}x_j^{(k)}, & i\neq j \\
0, & i= j 
\end{cases}
$$

其中，$x_i^{(k)}$ 是第$k$个要存储的向量的第$i$分量。

### 更新规则

Hopfield神经网络中的神经元的更新规则如下：

$$
x_{i} = \begin{cases}
1, & \sum_j w_{ij}x_j \ge b_i\\
-1, & \sum_j w_{ij}x_j < b_i
\end{cases}
$$

其中$𝑥_𝑖$是神经元$𝑖$的状态，$𝑏_𝑖$是神经元$𝑖$的阈值。根据GPT给出的更新过程，$b_i$统一取$0$即可。

网络状态的更新有同步和异步两种方式，同步更新同时更新部分或者所有神经元的状态，然而因为保证多个神经元同步的难度很大，实际中使用更多的是异步更新，即每次只更新一个神经元的状态。

### 有效性分析

为什么在上述的权重计算法和更新规则下，能够保证网络最终走到稳定状态（能量函数最小）？这似乎并不显然。从数学上应该可以证明这一策略的有效性。

首先，我们需要证明Hopfield网络在使用Hebbian学习规则后，所存储的模式是网络的稳定状态。这意味着，如果以存储的某一模式作为网络的初始状态，经过网络状态更新后，它仍然会保持在该模式。这一点可以通过计算验证。

其次，还需要证明对任意输入，都会收敛到某个稳定状态，这可以通过能量函数的单调递减性来证明。只要证明在每次状态更新后，能量函数$E$是单调递减的，又由于Hopfield网络的状态是有限的，所以能量函数$E$有界，那么不断递减的结果就只能是达到其下界——某个稳定状态。

能量函数的单调递减性同样可以计算验证。