# NLP-Beginner
自然语言处理入门教程



参考：[深度学习上手指南](https://github.com/nndl/nndl.github.io/blob/master/md/DeepGuide.md)



1. 文本分类，实现基于logistic/softmax regression的[文本分类](文本分类.md)

   1. 数据集：[Classify the sentiment of sentences from the Rotten Tomatoes dataset](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)
   2. 实现要求：NumPy
   3. 需要了解的知识点：

      1. 文本特征表示：Bag-of-Word，N-gram
      2. 分类器：logistic regression，损失函数、（随机）梯度下降、特征选择
      3. 数据集：训练集/验证集/测试集的划分
   4. 实验：分析不同的特征、损失函数、学习率对最终分类性能的影响
   5. 时间：一周
2. 文本分类Ⅱ，基于词向量和深度学习的文本分类
   1. 数据集：Classify the sentiment of sentences from the Rotten Tomatoes dataset
   2. 实现要求：Tensorflow
   3. 具体要求：
      1. tensorflow 重写写一遍.上一个问题的分类部分（重写的是分三类的问题）
      2. 把词用embedding的方式初始化。
         1. 随机embedding的初始化方式
         2. 用glove训练出来的文本初始
      3. 尝试shuffle 、batch、mini-batch  这三种方法。 
   4. 接下来是CNN、LSTM