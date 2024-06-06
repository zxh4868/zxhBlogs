### Problems with RNN

![image-20240518100219787](assets\image-20240518100219787.png)

#### Slow computation for long sequence

计算效率较低：RNN的计算是基于时间步展开的，每个时间步都需要以此计算(当前步依赖于上一步)，尤其是处理长序列文本时。

#### Vanishing or exploding gradients

梯度消失/爆炸问题：RNN在反向传播时，由于参数共享和多次连乘，容易出现梯度消失或梯度爆炸的问题，导致模型难以训练或无法收敛。

#### Difficulty in accessing information from long time ago

长期依赖问题：在处理长序列时难以捕捉到长期依赖关系(在一个长序列中，第一个token对最后一个token的贡献几乎为0)，只能有效利用较短的上下文信息

### Introducing the Transformer

![image-20240518100257016](assets\image-20240518100257016.png)

- Transformer总体分为两个部分：编码器和解码器，编码器的一些信息输入到了解码器

#### Input matrix(sequence, d_model)

![img](assets\d35a23e7b60e4dbd9e7dc00811778ce1.png)

#### Input Embedding

![image-20240518102025371](assets\image-20240518102025371.png)

1. 将原始句子分割为token
2. 将每一个token映射为一个数字（该token在我们训练集词汇表中的编号，比如YOUR在词汇表的位置是105）
3. 将这些数字(Input ID)映射为维度大小为5**12**的向量中

### 参考资料

https://www.mittrchina.com/news/detail/11157

