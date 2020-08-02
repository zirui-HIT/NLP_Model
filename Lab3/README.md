数据地址：[The Stanford Natural Language Inference (SNLI) Corpus](https://nlp.stanford.edu/projects/snli/)

由于训练集和GloVe预训练的embedding matrix过大（>100m），故未push到仓库

* pretreat.py：预处理，包括生成单词表和embedding matrix
* data.py：数据处理，包括加载数据、分词、填充等
* layer.py：ESIM的各个神经层
* module.py：整合layer中的各个神经层
* train.py：模型训练
* test.py：模型测试
* main.py：入口程序，可以在这里设置训练的参数、训练集和测试集

[博客总结](https://www.cnblogs.com/WDZRMPCBIT/p/13418634.html)