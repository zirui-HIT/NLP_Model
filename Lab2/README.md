题目地址：[电影评价的情感分析](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/overview)

### FNN
* 采用词袋模型
* 用PCA进行降维
* 1输入层+1全连接层+1输出层
* 全连接层神经元个数与单词数量相同
* 训练代数为50

### CNN
* 采用词向量+embedding
* embedding维数为32
* 1输入层+1卷积层+1汇聚层+输出层
* 卷积层out channel宽度为16，卷积核大小为5
* 汇聚层卷积核大小为2
* 采用批训练，批大小为1000
* 训练代数为20

### RNN
* 采用词向量+embedding
* embedding维数为16
* 1输入层+1隐藏层+1输出层
* 隐藏层神经元个数为40
* 采用LSTM
* 隐藏层后筛选有效维（[:最后一个非零位]）
* 采用批训练，批大小为1000
* 训练代数为20

[总结地址](https://www.cnblogs.com/WDZRMPCBIT/p/13399299.html)