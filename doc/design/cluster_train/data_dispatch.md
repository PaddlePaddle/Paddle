## 训练数据的存储和分发

### 流程介绍
生产环境中的训练数据集通常体积很大，并被存储在诸如Hadoop HDFS，Ceph，AWS S3之类的分布式存储之上。这些分布式存储服务通常会把数据切割成多个分片分布式的存储在多个节点之上。这样就可以在云端执行多种数据类计算任务，包括：

* 数据预处理任务
* Paddle训练任务
* 在线模型预测服务

<img src="src/paddle-cloud-in-data-center.png" width="500"/>

在上图中显示了在一个实际生产环境中的应用（人脸识别）的数据流图。生产环境的日志数据会通过实时流的方式（Kafka）和离线数据的方式（HDFS）存储，并在集群中运行多个分布式数据处理任务，比如流式数据处理（online data process），离线批处理（offline data process）完成数据的预处理，提供给paddle作为训练数据。用于也可以上传labeled data到分布式存储补充训练数据。在paddle之上运行的深度学习训练输出的模型会提供给在线人脸识别的应用使用。

### 训练数据的存储

选择GlusterFS作为训练数据的存储服务（后续的实现考虑HDFS）。

在Kubernetes上运行的不同的计算框架，可以通过Volume或PersistentVolume挂载存储空间到每个容器中。

在GlusterFS存储系统中的公开目录，需要保存一些预置的公开数据集（比如MNIST, BOW, imagenet数据集等），并且可以被提交的job直接使用。

### 上传训练文件

使用下面命令，可以把本地的训练数据上传到存储集群中，并指定上传数据的`dataset-name`：

```
paddle upload train_data.list "dataset-name"
```

其中`.list`文件描述了训练数据的文件和对应的label，对于图像类数据，`.list文件`样例如下，每一行包含了图片文件的路径和其label（用tab分隔开）：

```
./data/image1.jpg   1
./data/image2.jpg   5
./data/image3.jpg   2
./data/image4.jpg   5
./data/image5.jpg   1
./data/image6.jpg   8
...
```

对于文本类训练数据样例如下（机器翻译），一行中包含源语言，目标语言的文本（label）：

```
L&apos; inflation , en Europe , a dérapé sur l&apos; alimentation	Food : Where European inflation slipped up

L&apos; inflation accélérée , mesurée dans la zone euro , est due principalement à l&apos; augmentation rapide des prix de l&apos; alimentation .	The skyward zoom in food prices is the dominant force behind the speed up in eurozone inflation .
...
```

### 使用reader

用户在使用v2 API编写训练任务时，可以使用paddle内置的reader完成对GlusterFS存储中的训练数据的读取，返回文件中的各列，然后在调用`trainer.train()`时传入，完成训练数据的读取：

```python
reader = paddle.dist.reader("dataset-name")
trainer.train(reader, ...)
batch_reader = paddle.batch(paddle.dataset.mnist.train(), 128)
trainer.train(batch_reader, ...)
```

trainer.train内部会获取reader的内容：

```
def paddle.train(batch_reader):
  r = batch_reader() # create a iterator for one pass of data
  for batch in r:
    # train
```

这里面batch是含有128个data instance的mini-batch。每一个data instance会是一个tuple，tuple元素的顺序与`.list`文件文件中每一列的顺序是一致的。每一个data instance会是（raw_image_file_binary_data, label）。其中raw_image_file_binary_data是对应图像文件的没有解码的原始二进制数据，用户需要自己解码。label是文本类型（比如：“1“，”2“），这里用户需要的其实是整形，用户需要自己转换成整形。

### 实现reader

reader的实现需要考虑本地训练程序实现之后，可以不修改程序直接提交集群进行分布式训练。要达到这样的目标，需要实现下面的功能：

paddle会封装一个在集群中使用的reader: `paddle.dist.reader()`。在集群训练时需要使用这个reader指定要使用的数据集开始训练。用户的训练程序需要按照如下方式初始化reader：

```python
if os.getenv("PADDLE_TRAIN_LOCAL"):
  reader = my_local_reader("dataset-name")
else:
  reader = paddle.dist.reader("dataset-name")
```

用户训练程序提交到集群之后，集群会自动设置`PADDLE_TRAIN_LOCAL`环境变量，reader会被配置成集群训练的版本。其中`paddle.dist.reader()`需要从master的队列中获得需要开始执行的训练task，并找到对应的训练数据文件，开始训练任务。如果用户的训练数据源来自于其他服务，比如从集群中的Kafka，zeromq队列读取，也可以根据实际情况实现集群中运行的reader程序。

## TODO

### 支持将数据合并成内部的文件格式（key-value），方便sharding与顺序读取
### 支持用户自定义的数据预处理job
