# 情感分析

本教程源代码目录在[book/understand_sentiment](https://github.com/PaddlePaddle/book/tree/develop/06.understand_sentiment)， 初次使用请参考PaddlePaddle[安装教程](https://github.com/PaddlePaddle/book/blob/develop/README.cn.md#运行这本书)，更多内容请参考本教程的[视频课堂](http://bit.baidu.com/course/detail/id/177.html)。

## 背景介绍

在自然语言处理中，情感分析一般是指判断一段文本所表达的情绪状态。其中，一段文本可以是一个句子，一个段落或一个文档。情绪状态可以是两类，如（正面，负面），（高兴，悲伤）；也可以是三类，如（积极，消极，中性）等等。情感分析的应用场景十分广泛，如把用户在购物网站（亚马逊、天猫、淘宝等）、旅游网站、电影评论网站上发表的评论分成正面评论和负面评论；或为了分析用户对于某一产品的整体使用感受，抓取产品的用户评论并进行情感分析等等。表格1展示了对电影评论进行情感分析的例子：

| 电影评论       | 类别  |
| --------     | -----  |
| 在冯小刚这几年的电影里，算最好的一部的了| 正面 |
| 很不好看，好像一个地方台的电视剧     | 负面 |
| 圆方镜头全程炫技，色调背景美则美矣，但剧情拖沓，口音不伦不类，一直努力却始终无法入戏| 负面|
|剧情四星。但是圆镜视角加上婺源的风景整个非常有中国写意山水画的感觉，看得实在太舒服了。。|正面|

<p align="center">表格 1 电影评论情感分析</p>

在自然语言处理中，情感分析属于典型的**文本分类**问题，即把需要进行情感分析的文本划分为其所属类别。文本分类涉及文本表示和分类方法两个问题。在深度学习的方法出现之前，主流的文本表示方法为词袋模型BOW(bag of words)，话题模型等等；分类方法有SVM(support vector machine), LR(logistic regression)等等。

对于一段文本，BOW表示会忽略其词顺序、语法和句法，将这段文本仅仅看做是一个词集合，因此BOW方法并不能充分表示文本的语义信息。例如，句子“这部电影糟糕透了”和“一个乏味，空洞，没有内涵的作品”在情感分析中具有很高的语义相似度，但是它们的BOW表示的相似度为0。又如，句子“一个空洞，没有内涵的作品”和“一个不空洞而且有内涵的作品”的BOW相似度很高，但实际上它们的意思很不一样。

本章我们所要介绍的深度学习模型克服了BOW表示的上述缺陷，它在考虑词顺序的基础上把文本映射到低维度的语义空间，并且以端对端（end to end）的方式进行文本表示及分类，其性能相对于传统方法有显著的提升\[[1](#参考文献)\]。

## 模型概览
本章所使用的文本表示模型为卷积神经网络（Convolutional Neural Networks）和循环神经网络(Recurrent Neural Networks)及其扩展。下面依次介绍这几个模型。

### 文本卷积神经网络简介（CNN）

我们在[推荐系统](https://github.com/PaddlePaddle/book/tree/develop/05.recommender_system)一节介绍过应用于文本数据的卷积神经网络模型的计算过程，这里进行一个简单的回顾。

对卷积神经网络来说，首先使用卷积处理输入的词向量序列，产生一个特征图（feature map），对特征图采用时间维度上的最大池化（max pooling over time）操作得到此卷积核对应的整句话的特征，最后，将所有卷积核得到的特征拼接起来即为文本的定长向量表示，对于文本分类问题，将其连接至softmax即构建出完整的模型。在实际应用中，我们会使用多个卷积核来处理句子，窗口大小相同的卷积核堆叠起来形成一个矩阵，这样可以更高效的完成运算。另外，我们也可使用窗口大小不同的卷积核来处理句子，[推荐系统](https://github.com/PaddlePaddle/book/tree/develop/05.recommender_system)一节的图3作为示意画了四个卷积核，不同颜色表示不同大小的卷积核操作。

对于一般的短文本分类问题，上文所述的简单的文本卷积网络即可达到很高的正确率\[[1](#参考文献)\]。若想得到更抽象更高级的文本特征表示，可以构建深层文本卷积神经网络\[[2](#参考文献),[3](#参考文献)\]。

### 循环神经网络（RNN）

循环神经网络是一种能对序列数据进行精确建模的有力工具。实际上，循环神经网络的理论计算能力是图灵完备的\[[4](#参考文献)\]。自然语言是一种典型的序列数据（词序列），近年来，循环神经网络及其变体（如long short term memory\[[5](#参考文献)\]等）在自然语言处理的多个领域，如语言模型、句法解析、语义角色标注（或一般的序列标注）、语义表示、图文生成、对话、机器翻译等任务上均表现优异甚至成为目前效果最好的方法。

![rnn](./image/rnn.png)
<p align="center">
图1. 循环神经网络按时间展开的示意图
</p>

循环神经网络按时间展开后如图1所示：在第`$t$`时刻，网络读入第`$t$`个输入`$x_t$`（向量表示）及前一时刻隐层的状态值`$h_{t-1}$`（向量表示，`$h_0$`一般初始化为`$0$`向量），计算得出本时刻隐层的状态值`$h_t$`，重复这一步骤直至读完所有输入。如果将循环神经网络所表示的函数记为`$f$`，则其公式可表示为：

$$h_t=f(x_t,h_{t-1})=\sigma(W_{xh}x_t+W_{hh}h_{t-1}+b_h)$$

其中`$W_{xh}$`是输入到隐层的矩阵参数，`$W_{hh}$`是隐层到隐层的矩阵参数，`$b_h$`为隐层的偏置向量（bias）参数，`$\sigma$`为`$sigmoid$`函数。

在处理自然语言时，一般会先将词（one-hot表示）映射为其词向量（word embedding）表示，然后再作为循环神经网络每一时刻的输入`$x_t$`。此外，可以根据实际需要的不同在循环神经网络的隐层上连接其它层。如，可以把一个循环神经网络的隐层输出连接至下一个循环神经网络的输入构建深层（deep or stacked）循环神经网络，或者提取最后一个时刻的隐层状态作为句子表示进而使用分类模型等等。

### 长短期记忆网络（LSTM）

对于较长的序列数据，循环神经网络的训练过程中容易出现梯度消失或爆炸现象\[[6](#参考文献)\]。为了解决这一问题，Hochreiter S, Schmidhuber J. (1997)提出了LSTM(long short term memory\[[5](#参考文献)\])。

相比于简单的循环神经网络，LSTM增加了记忆单元`$c$`、输入门`$i$`、遗忘门`$f$`及输出门`$o$`。这些门及记忆单元组合起来大大提升了循环神经网络处理长序列数据的能力。若将基于LSTM的循环神经网络表示的函数记为`$F$`，则其公式为：

$$ h_t=F(x_t,h_{t-1})$$

`$F$`由下列公式组合而成\[[7](#参考文献)\]：
$$ i_t = \sigma{(W_{xi}x_t+W_{hi}h_{t-1}+W_{ci}c_{t-1}+b_i)} $$
$$ f_t = \sigma(W_{xf}x_t+W_{hf}h_{t-1}+W_{cf}c_{t-1}+b_f) $$
$$ c_t = f_t\odot c_{t-1}+i_t\odot tanh(W_{xc}x_t+W_{hc}h_{t-1}+b_c) $$
$$ o_t = \sigma(W_{xo}x_t+W_{ho}h_{t-1}+W_{co}c_{t}+b_o) $$
$$ h_t = o_t\odot tanh(c_t) $$
其中，`$i_t, f_t, c_t, o_t$`分别表示输入门，遗忘门，记忆单元及输出门的向量值，带角标的`$W$`及`$b$`为模型参数，`$tanh$`为双曲正切函数，`$\odot$`表示逐元素（elementwise）的乘法操作。输入门控制着新输入进入记忆单元`$c$`的强度，遗忘门控制着记忆单元维持上一时刻值的强度，输出门控制着输出记忆单元的强度。三种门的计算方式类似，但有着完全不同的参数，它们各自以不同的方式控制着记忆单元`$c$`，如图2所示：

![lstm](./image/lstm.png)
<p align="center">
图2. 时刻`$t$`的LSTM [7]
</p>

LSTM通过给简单的循环神经网络增加记忆及控制门的方式，增强了其处理远距离依赖问题的能力。类似原理的改进还有Gated Recurrent Unit (GRU)\[[8](#参考文献)\]，其设计更为简洁一些。**这些改进虽然各有不同，但是它们的宏观描述却与简单的循环神经网络一样（如图2所示），即隐状态依据当前输入及前一时刻的隐状态来改变，不断地循环这一过程直至输入处理完毕：**

$$ h_t=Recrurent(x_t,h_{t-1})$$

其中，`$Recrurent$`可以表示简单的循环神经网络、GRU或LSTM。

### 栈式双向LSTM（Stacked Bidirectional LSTM）

对于正常顺序的循环神经网络，`$h_t$`包含了`$t$`时刻之前的输入信息，也就是上文信息。同样，为了得到下文信息，我们可以使用反方向（将输入逆序处理）的循环神经网络。结合构建深层循环神经网络的方法（深层神经网络往往能得到更抽象和高级的特征表示），我们可以通过构建更加强有力的基于LSTM的栈式双向循环神经网络\[[9](#参考文献)\]，来对时序数据进行建模。

如图3所示（以三层为例），奇数层LSTM正向，偶数层LSTM反向，高一层的LSTM使用低一层LSTM及之前所有层的信息作为输入，对最高层LSTM序列使用时间维度上的最大池化即可得到文本的定长向量表示（这一表示充分融合了文本的上下文信息，并且对文本进行了深层次抽象），最后我们将文本表示连接至softmax构建分类模型。

![stacked_lstm](./image/stacked_lstm.jpg)
<p align="center">
图3. 栈式双向LSTM用于文本分类
</p>


## 数据集介绍

我们以[IMDB情感分析数据集](http://ai.stanford.edu/%7Eamaas/data/sentiment/)为例进行介绍。IMDB数据集的训练集和测试集分别包含25000个已标注过的电影评论。其中，负面评论的得分小于等于4，正面评论的得分大于等于7，满分10分。
```text
aclImdb
|- test
|-- neg
|-- pos
|- train
|-- neg
|-- pos
```
Paddle在`dataset/imdb.py`中提实现了imdb数据集的自动下载和读取，并提供了读取字典、训练数据、测试数据等API。

## 配置模型

在该示例中，我们实现了两种文本分类算法，分别基于[推荐系统](https://github.com/PaddlePaddle/book/tree/develop/05.recommender_system)一节介绍过的文本卷积神经网络，以及[栈式双向LSTM](#栈式双向LSTM（Stacked Bidirectional LSTM）)。我们首先引入要用到的库和定义全局变量：

```python
import paddle
import paddle.fluid as fluid
from functools import partial
import numpy as np

CLASS_DIM = 2
EMB_DIM = 128
HID_DIM = 512
BATCH_SIZE = 128
USE_GPU = False
```


### 文本卷积神经网络
我们构建神经网络`convolution_net`，示例代码如下。
需要注意的是：`fluid.nets.sequence_conv_pool` 包含卷积和池化层两个操作。

```python
def convolution_net(data, input_dim, class_dim, emb_dim, hid_dim):
emb = fluid.layers.embedding(
input=data, size=[input_dim, emb_dim], is_sparse=True)
conv_3 = fluid.nets.sequence_conv_pool(
input=emb,
num_filters=hid_dim,
filter_size=3,
act="tanh",
pool_type="sqrt")
conv_4 = fluid.nets.sequence_conv_pool(
input=emb,
num_filters=hid_dim,
filter_size=4,
act="tanh",
pool_type="sqrt")
prediction = fluid.layers.fc(
input=[conv_3, conv_4], size=class_dim, act="softmax")
return prediction
```

网络的输入`input_dim`表示的是词典的大小，`class_dim`表示类别数。这里，我们使用[`sequence_conv_pool`](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/trainer_config_helpers/networks.py) API实现了卷积和池化操作。

### 栈式双向LSTM

栈式双向神经网络`stacked_lstm_net`的代码片段如下：

```python
def stacked_lstm_net(data, input_dim, class_dim, emb_dim, hid_dim, stacked_num):

emb = fluid.layers.embedding(
input=data, size=[input_dim, emb_dim], is_sparse=True)

fc1 = fluid.layers.fc(input=emb, size=hid_dim)
lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim)

inputs = [fc1, lstm1]

for i in range(2, stacked_num + 1):
fc = fluid.layers.fc(input=inputs, size=hid_dim)
lstm, cell = fluid.layers.dynamic_lstm(
input=fc, size=hid_dim, is_reverse=(i % 2) == 0)
inputs = [fc, lstm]

fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')

prediction = fluid.layers.fc(input=[fc_last, lstm_last],
size=class_dim,
act='softmax')
return prediction
```
以上的栈式双向LSTM抽象出了高级特征并把其映射到和分类类别数同样大小的向量上。`paddle.activation.Softmax`函数用来计算分类属于某个类别的概率。

重申一下，此处我们可以调用`convolution_net`或`stacked_lstm_net`的任何一个。我们以`convolution_net`为例。

接下来我们定义预测程序（`inference_program`）。预测程序使用`convolution_net`来对`fluid.layer.data`的输入进行预测。

```python
def inference_program(word_dict):
data = fluid.layers.data(
name="words", shape=[1], dtype="int64", lod_level=1)

dict_dim = len(word_dict)
net = convolution_net(data, dict_dim, CLASS_DIM, EMB_DIM, HID_DIM)
return net
```

我们这里定义了`training_program`。它使用了从`inference_program`返回的结果来计算误差。我们同时定义了优化函数`optimizer_func`。

因为是有监督的学习，训练集的标签也在`paddle.layer.data`中定义了。在训练过程中，交叉熵用来在`paddle.layer.classification_cost`中作为损失函数。

在测试过程中，分类器会计算各个输出的概率。第一个返回的数值规定为 损耗(cost)。

```python
def train_program(word_dict):
prediction = inference_program(word_dict)
label = fluid.layers.data(name="label", shape=[1], dtype="int64")
cost = fluid.layers.cross_entropy(input=prediction, label=label)
avg_cost = fluid.layers.mean(cost)
accuracy = fluid.layers.accuracy(input=prediction, label=label)
return [avg_cost, accuracy]


def optimizer_func():
return fluid.optimizer.Adagrad(learning_rate=0.002)
```

## 训练模型

### 定义训练环境

定义您的训练是在CPU上还是在GPU上：


```python
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
```

### 定义数据提供器

下一步是为训练和测试定义数据提供器。提供器读入一个大小为 BATCH_SIZE的数据。paddle.dataset.imdb.train 每次会在乱序化后提供一个大小为BATCH_SIZE的数据，乱序化的大小为缓存大小buf_size。

注意：读取IMDB的数据可能会花费几分钟的时间，请耐心等待。

```python
print("Loading IMDB word dict....")
word_dict = paddle.dataset.imdb.word_dict()

print ("Reading training data....")
train_reader = paddle.batch(
paddle.reader.shuffle(
paddle.dataset.imdb.train(word_dict), buf_size=25000),
batch_size=BATCH_SIZE)
```

### 构造训练器(trainer)
训练器需要一个训练程序和一个训练优化函数。

```python
trainer = fluid.Trainer(
train_func=partial(train_program, word_dict),
place=place,
optimizer_func=optimizer_func)
```

### 提供数据

`feed_order`用来定义每条产生的数据和`paddle.layer.data`之间的映射关系。比如，`imdb.train`产生的第一列的数据对应的是`words`这个特征。

```python
feed_order = ['words', 'label']
```

### 事件处理器

回调函数event_handler在一个之前定义好的事件发生后会被调用。例如，我们可以在每步训练结束后查看误差。

```python
# Specify the directory path to save the parameters
params_dirname = "understand_sentiment_conv.inference.model"

def event_handler(event):
if isinstance(event, fluid.EndStepEvent):
print("Step {0}, Epoch {1} Metrics {2}".format(
event.step, event.epoch, map(np.array, event.metrics)))

if event.step == 10:
trainer.save_params(params_dirname)
trainer.stop()
```

### 开始训练

最后，我们传入训练循环数（num_epoch）和一些别的参数，调用 trainer.train 来开始训练。

```python
trainer.train(
num_epochs=1,
event_handler=event_handler,
reader=train_reader,
feed_order=feed_order)
```

## 应用模型

### 构建预测器

传入`inference_program`和`params_dirname`来初始化一个预测器, `params_dirname`用来存放训练过程中的各个参数。

```python
inferencer = fluid.Inferencer(
inference_program, param_path=params_dirname, place=place)
```

### 生成测试用输入数据

为了进行预测，我们任意选取3个评论。请随意选取您看好的3个。我们把评论中的每个词对应到`word_dict`中的id。如果词典中没有这个词，则设为`unknown`。
然后我们用`create_lod_tensor`来创建细节层次的张量。

```python
reviews_str = [
'read the book forget the movie', 'this is a great movie', 'this is very bad'
]
reviews = [c.split() for c in reviews_str]

UNK = word_dict['<unk>']
lod = []
for c in reviews:
lod.append([word_dict.get(words, UNK) for words in c])

base_shape = [[len(c) for c in lod]]

tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
```

## 应用模型

现在我们可以对每一条评论进行正面或者负面的预测啦。

```python
results = inferencer.infer({'words': tensor_words})

for i, r in enumerate(results[0]):
print("Predict probability of ", r[0], " to be positive and ", r[1], " to be negative for review \'", reviews_str[i], "\'")

```


## 总结

本章我们以情感分析为例，介绍了使用深度学习的方法进行端对端的短文本分类，并且使用PaddlePaddle完成了全部相关实验。同时，我们简要介绍了两种文本处理模型：卷积神经网络和循环神经网络。在后续的章节中我们会看到这两种基本的深度学习模型在其它任务上的应用。


## 参考文献
1. Kim Y. [Convolutional neural networks for sentence classification](http://arxiv.org/pdf/1408.5882)[J]. arXiv preprint arXiv:1408.5882, 2014.
2. Kalchbrenner N, Grefenstette E, Blunsom P. [A convolutional neural network for modelling sentences](http://arxiv.org/pdf/1404.2188.pdf?utm_medium=App.net&utm_source=PourOver)[J]. arXiv preprint arXiv:1404.2188, 2014.
3. Yann N. Dauphin, et al. [Language Modeling with Gated Convolutional Networks](https://arxiv.org/pdf/1612.08083v1.pdf)[J] arXiv preprint arXiv:1612.08083, 2016.
4. Siegelmann H T, Sontag E D. [On the computational power of neural nets](http://research.cs.queensu.ca/home/akl/cisc879/papers/SELECTED_PAPERS_FROM_VARIOUS_SOURCES/05070215382317071.pdf)[C]//Proceedings of the fifth annual workshop on Computational learning theory. ACM, 1992: 440-449.
5. Hochreiter S, Schmidhuber J. [Long short-term memory](http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf)[J]. Neural computation, 1997, 9(8): 1735-1780.
6. Bengio Y, Simard P, Frasconi P. [Learning long-term dependencies with gradient descent is difficult](http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf)[J]. IEEE transactions on neural networks, 1994, 5(2): 157-166.
7. Graves A. [Generating sequences with recurrent neural networks](http://arxiv.org/pdf/1308.0850)[J]. arXiv preprint arXiv:1308.0850, 2013.
8. Cho K, Van Merriënboer B, Gulcehre C, et al. [Learning phrase representations using RNN encoder-decoder for statistical machine translation](http://arxiv.org/pdf/1406.1078)[J]. arXiv preprint arXiv:1406.1078, 2014.
9. Zhou J, Xu W. [End-to-end learning of semantic role labeling using recurrent neural networks](http://www.aclweb.org/anthology/P/P15/P15-1109.pdf)[C]//Proceedings of the Annual Meeting of the Association for Computational Linguistics. 2015.

<br/>
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" href="http://purl.org/dc/dcmitype/Text" property="dct:title" rel="dct:type">本教程</span> 由 <a xmlns:cc="http://creativecommons.org/ns#" href="http://book.paddlepaddle.org" property="cc:attributionName" rel="cc:attributionURL">PaddlePaddle</a> 创作，采用 <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">知识共享 署名-相同方式共享 4.0 国际 许可协议</a>进行许可。
