数据来源：[全唐诗](./test/poetryFromTang.txt)

embedding后用LSTM进行拟合，每一个time产生的输出就看做对当前位置上词的特性的提取，然后传入一个全连接层，产生相应的词向量。

对输出和embedding后的原句求MSE，然后反向传播。

每个句子前后插入BOS和EOS作为句子的开始、结束标记。

由于未考虑句子的长度，所以生成的应该叫词XD