#### 极简归纳

Attention并非一个具体的网络或者层，它是一种“机制”，一种在encoder-decoder结构中计算上下文向量的思路。Transformer则是一个具体的网络，其中一个重要的部分是self-attention，这里的self-attention可以当成一个模块。

要搞明白attention的提出，可以沿着“翻译器”的发展史来学习。首先出现了RNN网络，接下来为了实现不等长输入输出，出现了encoder-decoder结构和上下文向量，再之后为了拓展上下文向量的记录能力，出现了attention思想。attention在具体实现中还有很多细节问题可以讨论。

建立在上述内容的基础上，transformer是一个非常复杂的模型，它对attention思想的具体实现被称为self-attention结构。

#### 参考 

从RNN到Encoder-Decoder，再到Attention：https://zhuanlan.zhihu.com/p/28054589

对于Attention中的权重$a_{ij}$如何计算，给出了细节；之后还给出了attention的更多种类：https://zhuanlan.zhihu.com/p/380892265

上文中关于global attention和local attention说的不够清楚，这个清楚一些：https://blog.csdn.net/weixin_40871455/article/details/85007560

transformer讲解，侧重attention的矩阵乘法：https://zhuanlan.zhihu.com/p/311156298

self-attention的内在原理讲解：https://www.zhihu.com/column/p/410776234

同样是self-attention，内容不多，但有很多好看的无水印图：https://blog.csdn.net/Lamours/article/details/125192046

transformer的另一个讲解，细节很多：https://zhuanlan.zhihu.com/p/338817680

更多关于transformer的图：https://blog.csdn.net/weixin_44305115/article/details/101622645

Transformer全过程详解：《BERT基础教程：Transformer大模型实战》



# Transformer

机器翻译（Machine Translation, MT）的发展可以追溯到20世纪40年代，经历了几个主要阶段。早期的机器翻译尝试基于规则，依靠语言学家编写的大量语法和词汇规则来转换源语言到目标语言。随着计算能力的增强和数字文本数据的爆炸性增长，基于统计的方法开始流行。这种方法不再依赖于硬编码的语言规则，而是通过分析大量的双语文本语料库（称为平行语料库）来学习词汇之间的统计关系。

随着深度学习技术的兴起，基于神经网络的机器翻译开始发展，特别是循环神经网络（RNN）和长短期记忆网络（LSTM）被用来处理序列到序列的任务。NMT能够学习从源语言到目标语言的端到端映射，而不需要复杂的特征工程。这种方法又统一采用了编码器-解码器（encoder-decoder）架构。这种架构是为了处理序列到序列（Seq2Seq）的任务，可以理解为编码器将输入翻译成某种“机器语言”，解码器再将“机器语言”翻译出需要的输出。

Transformer模型同样处理序列到序列的任务，当然也包括机器翻译。大体上讲，它依然遵循encoder-decoder架构，但编码解码过程与过去完全不同。接下来我们详细描述Transformer的过程。摆一个全流程图镇楼：

![image-20240401200001816](D:\GithubRepos\notes_about_datascience\note\img\image-20240401200001816.png)

### 准备工作

#### 词向量嵌入

将一个句子拆解成多个token（一个token可以看作是一个“词素”，它未必是一个词，但在理解时可以等同视之），每个token转变成一个向量，设其维数为$d$。那么一个句子（作为一个样本）实际上表示为单词数×d的矩阵。为了方便，将单词数记为$c$。

#### 位置编码

当机器见到句子矩阵的时候，矩阵的行与行之间实际上是相互独立的。但作为语言，词汇的顺序非常重要，因此除了句子矩阵还得把单词的顺序送进去作为输入。这就是位置编码的作用，每个单词的位置编码是一个长度与嵌入维度相同的向量。

“Attention Is All You Need”的作者使用了正弦函数来计算位置编码：
$$
P(pos, 2i) = \sin \left( \frac{pos}{10000^{\frac{2i}{d}}} \right) \\
P(pos, 2i+1) = \cos \left( \frac{pos}{10000^{\frac{2i}{d}}} \right) \\
$$
其中pos表示该词在句子中的位置（从0开始计数），$i$是分量编号。

一个句子经过位置编码，得到的也是一个$c\times d$的矩阵。

### 编码器Encoder

Transformer中的编码器不止一个，而是由一组N个编码器串联而成，一个编码器的输出作为下一个编码器的输入。每一个编码器的构造都是相同的，并且包含两个部分：多头注意力层和前馈网络层。

#### 自注意力机制

根据输入矩阵，分别计算查询（query）矩阵Q、键（key）矩阵K，以及值（value）矩阵V三个矩阵。

计算注意力矩阵
$$
Z = \text{softmax} \left( \frac{QK^T}{\sqrt{d}} \right) V
$$
其中softmax函数是对行计算，也即其结果每一行的和为1。

#### 多头注意力层

每一个(Q, K, V)组可以看作是从一个角度来衡量词之间的相关性，如同CNN的卷积要进行很多次来捕捉不同的特征，计算注意力也会进行很多次，来提高注意力矩阵的准确性。

假设我们有8个注意力矩阵，即Z~1~到Z~8~，那么可以直接将所有的注意力头（注意力矩阵）串联起来，并将结果乘以一个新的权重矩阵$W_0$，从而得出最终的注意力矩阵，也即Multi-head attention = Concatenate(Z~1~,...,Z~8~) W~0~

#### 叠加与归一化

对应图中”Add & Norm“部分，叠加和归一组件实际上包含一个残差连接与层的归一化。层的归一化可以防止每层的值剧烈变化，从而提高了模型的训练速度。

#### 前馈网络层

前馈网络由两个有ReLU激活函数的全连接层组成。前馈网络的参数在句子的不同位置上是相同的，但在不同的编码器模块上是不同的。

### 解码器Decoder
