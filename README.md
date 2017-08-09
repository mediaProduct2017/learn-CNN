# learn-CNN

Convolutional neural network (CNN)的优势在于可以抓住不同feature之间的排列顺序。

对于二维图像识别，CNN能抓住横向的及纵向的像素点之间的相邻关系，把相邻关系反映在最终的模型中。

对于自然语言处理，当word vector用一维向量表示时，一维的单词排列顺序再加上一维的向量表示，就构成了一个二维的输入，和二维图像识别本质上是类似的，CNN既能抓住单词排列顺序的信息，也能抓住word vector的信息，把这两部分信息都反映在最终的模型中。

通过横向的和纵向的多次扫描，像素信息的矩阵越来越小，到最后某一个维度上的长度变成1，一个二维的矩阵转变为一个一维的向量，最终这个一维向量映射为单一类别（vanilla neural network）。

对于自然语言处理，扫描时一般只做横向扫描，也就是说，word vector不再做切割扫描，最简单的扫描方式是，2个word（或2个字）扫描一次，每次移动1个word（或1个字），逐次扫描过去。得到新的一层neuron后，再按同样的方法扫描过去。

可以认为，CNN是一种实用的Tree recursive neural network，包含着specific assumption。
