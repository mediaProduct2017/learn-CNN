# learn-CNN

Convolutional neural network (CNN)的优势在于可以抓住不同feature之间的排列顺序。

对于二维图像识别，CNN能抓住横向的及纵向的像素点之间的相邻关系，把相邻关系反映在最终的模型中。CNN会使用一个一定大小的filter，这个filter扫描图片的各个地方，通过线性和relu函数转变为一个值或者一个向量，但是，线性组合所用的weights是不变的，因为这个filter扫过的像素，它们的相邻关系都是一样的。

通过横向的和纵向的多次扫描，像素信息的矩阵越来越小，到最后某一个维度上的长度变成1，一个二维的矩阵转变为一个一维的向量，最终这个一维向量映射为单一类别（vanilla neural network）。实际上，一个多维的conv net转变为一维的fully connected layer，并不是一个个维度压缩的，而是展开成一个大的vector以后，直接转变为一维fully connected layer的。就好像simple neural network for image classification，input虽然是二维或者三维图像（RGB），但在给到neural net时，是展成一个大的vector来给的。

对于自然语言处理，当word vector用一维向量表示时，一维的单词排列顺序再加上一维的向量表示，就构成了一个二维的输入，和二维图像识别本质上是类似的，CNN既能抓住单词排列顺序的信息，也能抓住word vector的信息，把这两部分信息都反映在最终的模型中。不同的是，对于word vetor这个维度，实际上是没有顺序信息的，就好像图像的RGB三个图层，也是没有顺序信息的。所以，对于自然语言处理，扫描时一般只做横向扫描，也就是说，word vector不再做切割扫描，最简单的扫描方式是，2个word（或2个字）扫描一次，每次移动1个word（或1个字），逐次扫描过去。得到新的一层neuron后，再按同样的方法扫描过去。

可以认为，CNN是一种实用的Tree recursive neural network，包含着specific assumption。

对于大的data set，往往分批装入内存，然后用程序语言来处理。在cost function优化过程中，之所以用mini-batch的方法，也是因为内存不够，如果还用full-batch，在每一轮优化的过程中，都要不断把新的数据加载到内存中，覆盖掉之前的内存中的数据，在下一轮中，又要循环加载被覆盖掉的数据，这样的话，计算的效率太低。同时要注意的是，分批的时候一定要随机化，否则的话，如果某一批都是某个类别的话，针对该批的mini-batch training就没有太大的意义。

Explore data包括展示data某一维的histogram分布，或者是不同维度之间的相关关系（用散点图就能简单的展示）。必要的话，还可以针对不同种类样本画出各个维度的box plot，或者通过PCA进行低维的可视化。

Data normalization包括必要的话取对数并做outlier detection，以及减去均值除以标准差的归一化处理。归一化处理时有可能是以0为均值、以1为标准差的normal，也有可能是truncated normal，同样是以0位均值、以1为标准差，但是2以外的数值就不要了（其实常见于weights的初始化）。如果原始数据就都是正值，也有可能简单除以数据的最大值，或者除以数据可能的最大值，或者除以范围（数据的最大值-最小值）。Data normalization还包括one-hot encode for target variable.

在数据分析的noetbook中，经常设置check point，比如保存好数据后，下一个cell可以设置check point，简单的重新载入数据，如果重新启动可以从这一个cell开始。在训练好model之后，也经常会保存相应的参数，再设置一个check point.

Note: None for shapes in TensorFlow allow for a dynamic size. 也即是说，在TF placeholder中，None在具体传参的时候可以是任意值。

What's the difference between tf.placeholder and tf.Variable (stack overflow)?

In short, you use tf.Variable for trainable variables such as weights and biases for your model. tf.placeholder is used to feed actual training examples. Your tf.Variable will be trained (modified) during training.

## Convolution and max pooling layer

对于图片，input是三维的，第三维是RGB三个图层，filter是三维的，这三维对应图片的三个维度，第三个维度一般取3，也就是说，在RGB图层维度上，是不做切割的。三维的filter与四维的weights是对应的，weights的第四维是convolution layer的深度。filter strides也是四维的。In TensorFlow, strides is an array of 4 elements; the first element in this array indicates the stride for batch and last element indicates stride for features（也就是RGB三个图层）. It's good practice to remove the batches or features you want to skip from the data set rather than use a stride to skip them. You can always set the first and last element to 1 in strides in order to use all batches and features.

从input到convolution layer，和普通的neural net的input到hidden layer有神似，普通的neural net的input是把二维或者三维的图片展成一个大的向量，这里，input是把filter的三维input展成一个大的向量，然后通过线性变换对应到convolution layer上，对于一个filter，convolution layer是一个向量，大小是convolution layer的深度，其实相当于普通的neural net的hidden layer，convolution layer的深度就对应着hidden layer的node number.

对于convolution layer，线性变换之后，还要使用relu函数进行activate，比如用tf.nn.relu().

convolution layer从深度的一维到三维的过程，不过是filter在横纵两个维度上依次扫描罢了。在实际的应用中，convolution layer常常输出一个四维的tensor，第一维是batch size，就是用来训练的有多少个batch，接下来的三个维度依次是横纵和深度。

max pooling是一个降低模型复杂度，减少参数的过程，pooling的ksize一般也是四维，第一维batch pooling一般选1，第四维feature pooling一般也选1，只在横纵两个维度上做max pooling。pooling的strides的大小一般和pooling的大小是一致的。

如果数据量很大，可以不用pooling。

Convolution and max pooling layer有可能不只一个，有可能有多个，layer的深度在不断发生变化，每一层layer可能在提取不同的feature.

## Convolution layer的维度变换公式

How to determine the dimensions of the output based on the input size and the filter size (shown below). You'll use this to determine what the size of your filter should be.

对于SAME padding，P=1；对于VALID padding，P=0.

new_height = (input_height - filter_height + 2 * P)/S + 1

new_width = (input_width - filter_width + 2 * P)/S + 1

TensorFlow uses the following equation for 'SAME' vs 'VALID'

需要注意的是，对于SAME Padding，最终的convolution layer的长宽与filter的大小无关。

SAME Padding, the output height and width are computed as:

out_height = ceil(float(in_height) / float(strides[1]))

out_width = ceil(float(in_width) / float(strides[2]))

VALID Padding, the output height and width are computed as:

out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))

out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))

## Flatten layer

Flatten layer把convolution layer的输出从四维变成二维，第一维依旧是batch size，第二维其实是把convolution layer的后三个维度展开成一个大的向量。

## Fully connected layer

需要选择hidden layer的node number，与普通neural net的input的hidden layer的变化无异，就是线性变换，变换后需要用activation，比如relu activatoin。

在实际应用中，fully connected layer也可能有多层。

## Output layer

就是对fully connected layer做一个线性变换。

在tensorflow中，Activation, softmax, or cross entropy should not be applied to this step。softmax和cross entropy是针对output layer的输出进行的，在后面的步骤中进行，而且一般直接使用tensorflow所带的高级函数。

## Dropout

dropout是防止overfitting的一种办法，随机干掉一定比例的node，在某一个batch的拟合中，减少需要拟合的参数。keep probability参数越小，干掉的node越多，防止overfitting的作用越强。

dropout可以加在max pooling layer之后，也可以加在fully connected layer之后。一种常见的处理是，只加在最后一层fully connected layer之后。

dropout一般0.5左右效果最佳，可以在0.25-0.75之间尝试，如果是简单项目，调的不用那么细，0.5就行了。

For validation test set, use a keep probability of 1.0 to calculate the loss and validation accuracy.

## Hyperparameters

Set epochs to the number of iterations until the network stops learning or start overfitting

Set batch_size to the highest number that your machine has memory for. Most people set them to common sizes of memory. The larger, the better. For example, 256, or 128.

Set keep_probability to the probability of keeping a node using dropout

## Train on a single batch and fully train the model

the number of iterations设好之后，在每一个iteration之下，可以只有一批数据来训练，也可以是多批数据（多个mini batch），常用的一种方法是，先用一批数据，调整其他参数，但accuracy达到一定标准之后，再使用多批数据来训练。

Refercences:

[Deep learning and convolutional neural network](http://neuralnetworksanddeeplearning.com/chap6.html)

[Generative Adversarial Networks](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Generative-Adversarial-Networks)
