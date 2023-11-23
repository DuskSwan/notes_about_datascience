## 极简归纳

Attention并非一个具体的网络或者层，它是一种“机制”，一种在encoder-decoder结构中计算上下文向量的思路。Transformer则是一个具体的网络，其中一个重要的部分是self-attention，这里的self-attention可以当成一个模块。

要搞明白attention的提出，可以沿着“翻译器”的发展史来学习。首先出现了RNN网络，接下来为了实现不等长输入输出，出现了encoder-decoder结构和上下文向量，再之后为了拓展上下文向量的记录能力，出现了attention思想。attention在具体实现中还有很多细节问题可以讨论。

建立在上述内容的基础上，transformer是一个非常复杂的模型，它对attention思想的具体实现被称为self-attention结构。

## 参考 

从RNN到Encoder-Decoder，再到Attention：https://zhuanlan.zhihu.com/p/28054589

对于Attention中的权重$a_{ij}$如何计算，给出了细节；之后还给出了attention的更多种类：https://zhuanlan.zhihu.com/p/380892265

上文中关于global attention和local attention说的不够清楚，这个清楚一些：https://blog.csdn.net/weixin_40871455/article/details/85007560

transformer讲解，侧重attention的矩阵乘法：https://zhuanlan.zhihu.com/p/311156298

self-attention的内在原理讲解：https://www.zhihu.com/column/p/410776234

同样是self-attention，内容不多，但有很多好看的无水印图：https://blog.csdn.net/Lamours/article/details/125192046

transformer的另一个讲解，细节很多：https://zhuanlan.zhihu.com/p/338817680

更多关于transformer的图：https://blog.csdn.net/weixin_44305115/article/details/101622645
