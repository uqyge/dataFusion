# Identification of missing input distributions with an inverse multi-modal Polynomial Chaos approach based on scarce data

# 引言
最近读到 [*Identification of missing input distributions with an inverse multi-modal Polynomial Chaos approach based on scarce data*](https://www.sciencedirect.com/science/article/pii/S0266892021000229)这篇关于稀疏数据下的参数反演的文章。感觉普适性很强，所以利用python对文章的主要思想做一个实现。

文章的核心方法可以由图1很好的表述。
![](https://i.niupic.com/images/2022/03/29/9XQq.png)

整个参数反演的算法过程依图1分别包含上中下三条路径。
- 最上的路径A表示实验中数据的采集，与统计分析。
- 中间的路径表示了不确定性由输入参数经过数学模型的传播过程。
- 通过对比实验数据与计算结果，文章建立了一个针对未知输入参数的反演优化过程。
接下来我们尝试将三个过程分别利用python来做一个简单实践。

在本实验中，我们假定所研究的问题的数学模型已知，不失一般性，我们设模型表达式为，
$y = f(x1,x2)=2(x1+x2)$,
且已知$x1$满足标准正态分布，而$x2$也舒服正态分布，可是参数未知。本示例的最终目标就是通过不确定性传播与统计分析，结合反向优化算法，确定$x2$的均值$\mu$,与方差$\sigma$。

# 实验数据