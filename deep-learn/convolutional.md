# 卷积原理是案例

## 一、概念

卷积神经网络（Convolutional Neural Network, CNN）是一种前馈神经网络，对于大型图像处理能够有出色表现。

> 在以前学习监督学习和非监督学习算法的时候，认为``CNN``、``RNN``等神经网络结构是很了不起的知识，现在看来也只是另外一种算法的实现而已，就类似于监督学习中的``SVM``、``DecisionTree``，也可以看作一个分类器，不过用神经网络结构来实现能够更加高效简单。

## 二、介绍

### 2.1 简单神经网络单元

- 单个神经元

    一个简单的神经网络单元结构如下：

    ![](./imgs/neural-cell.png)

    用公式表达：

    ![](./imgs/neural-cell-1.png)

    熟悉监督学习的小伙伴就会发现，此公式与Logistic回归模型公式很类似。请思考五秒......

    对没错，监督学习算法可以用简单的神经元来实现。那么扩散一下，监督学习和非监督学习中的各类算法其实都可以用不同的神经网络来实现，而且在编码上来讲要更简单，更简单，更简单。

- 神经网络层

    当多个神经元联合组成分层结构，就形成了网络结构，一个简单的包含一层隐藏层网络结构如下：

    ![](./imgs/nerual-with-hidden-layer.png)

    > Layer L1为输入层，Layer L2为隐藏层，Layer L3为输出层

    神经网络的训练方法和监督学习简单算法中大同小异，只不过其结构相对复杂而已，一般采用梯度下降+链式求导法则，专业术语就是**反向传播**。

### 2.2 卷积神经网络介绍

- 什么是卷积

    已有paper和大牛解释过，我就不在赘述，直接奉上我的膝盖：[对卷积神经网络直观的解释](https://www.zhihu.com/question/39022858/answer/224446917)


***

假如有一个``1000*1000``的图像，那么输入层就是``1000000``维度，如果隐含层和输入层维度一样，而且还是全连接，那么根据以上对普通神经元的介绍中，参数个数就是：``1000000*1000000=10^12``，有木有感觉很可怕，这么多参数根本没法训练，就算有足够的的数据，也很难训练到位，所以就需要以下方法。

- 局部感知

    为了减少参数个数，于是就诞生了局部感知器。少废话，先看图：

    ![](./imgs/neural-part.jpg)

    左边就是每个神经单元都需要扫描整张图，所以需要``10^12``个参数；右边是每个神经单元只需要负责图片中的部分位置，假如每个神经元只需要负责``10*10``大小区域，如此，输入层和隐藏层的参数个数就为``1000000*100=10^8``(此时的``1000000``为隐藏层的神经元数量，``100``为每个神经元对应的参数数量)，这样参数个数就减少为原来的万分之一。

    有木有很棒，然而，``10^8``个参数还是太多了。

    

- 参数共享

    为了进一步减少参数数量，于是就需要有**权值共享**，


> 以上是卷积神经网络层中优化算法，我们不需要手动实现，只需要调整参数改变对应策略就行。

***

参考链接
- [wiki-卷积神经网络](https://zh.wikipedia.org/wiki/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C#cite_note-deeplearning-1)

- [cs231n-Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/)

- [卷积神经网络](https://blog.csdn.net/stdcoutzyx/article/details/41596663)