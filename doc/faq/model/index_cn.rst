#########
模型配置
#########

..  contents::

1. 出现 :code:`Duplicated layer name` 错误怎么办
--------------------------------------------------

出现该错误的原因一般是用户对不同layer的参数 :code:`name` 设置了相同的取值。遇到该错误时，先找出参数 :code:`name` 取值相同的layer，然后将这些layer的参数 :code:`name` 设置为不同的值。

2. :code:`paddle.layer.memory` 的参数 :code:`name` 如何使用
-------------------------------------------------------------

* :code:`paddle.layer.memory` 用于获取特定layer上一时间步的输出，该layer是通过参数 :code:`name` 指定，即，:code:`paddle.layer.memory` 会关联参数 :code:`name` 取值相同的layer，并将该layer上一时间步的输出作为自身当前时间步的输出。

* PaddlePaddle的所有layer都有唯一的name，用户通过参数 :code:`name` 设定，当用户没有显式设定时，PaddlePaddle会自动设定。而 :code:`paddle.layer.memory` 不是真正的layer，其name由参数 :code:`memory_name` 设定，当用户没有显式设定时，PaddlePaddle会自动设定。:code:`paddle.layer.memory` 的参数 :code:`name` 用于指定其要关联的layer，需要用户显式设定。

3. 两种使用 drop_out 的方法有何区别
------------------------------------

* 在PaddlePaddle中使用dropout有两种方式

  * 在相应layer的 :code:`layer_atter` 设置 :code:`drop_rate`，以 :code:`paddle.layer.fc` 为例，代码如下：

  ..  code-block:: python

      fc = paddle.layer.fc(input=input, layer_attr=paddle.attr.ExtraLayerAttribute(drop_rate=0.5))

  * 使用 :code:`paddle.layer.dropout`，以 :code:`paddle.layer.fc` 为例，代码如下：

  ..  code-block:: python

      fc = paddle.layer.fc(input=input)
      drop_fc = paddle.layer.dropout(input=fc, dropout_rate=0.5)

* :code:`paddle.layer.dropout` 实际上使用了 :code:`paddle.layer.add_to`，并在该layer里采用第一种方式设置 :code:`drop_rate` 来使用dropout的。这种方式对内存消耗较大。

* PaddlePaddle在激活函数里实现dropout，而不是在layer里实现。

* :code:`paddle.layer.lstmemory`、:code:`paddle.layer.grumemory`、:code:`paddle.layer.recurrent` 不是通过一般的方式来实现对输出的激活，所以不能采用第一种方式在这几个layer里设置 :code:`drop_rate` 来使用dropout。若要对这几个layer使用dropout，可采用第二种方式，即使用 :code:`paddle.layer.dropout`。

4. 不同的 recurrent layer 的区别
----------------------------------
以LSTM为例，在PaddlePaddle中包含以下 recurrent layer：

* :code:`paddle.layer.lstmemory`
* :code:`paddle.networks.simple_lstm`
* :code:`paddle.networks.lstmemory_group`
* :code:`paddle.networks.bidirectional_lstm`

按照具体实现方式可以归纳为2类：

1. 由 recurrent_group 实现的 recurrent layer：

  * 用户在使用这一类recurrent layer时，可以访问由recurrent unit在一个时间步内计算得到的中间值（例如：hidden states, memory cells等）；
  * 上述的 :code:`paddle.networks.lstmemory_group` 是这一类的 recurrent layer ；

2. 将recurrent layer作为一个整体来实现：

  * 用户在使用这一类recurrent layer，只能访问它们的输出值；
  * 上述的 :code:`paddle.networks.lstmemory_group` 、 :code:`paddle.networks.simple_lstm` 和 :code:`paddle.networks.bidirectional_lstm` 属于这一类的实现；

将recurrent layer作为一个整体来实现， 能够针对CPU和GPU的计算做更多优化， 所以相比于recurrent group的实现方式， 第二类 recurrent layer 计算效率更高。 在实际应用中，如果用户不需要访问LSTM的中间变量，而只需要获得recurrent layer计算的输出，我们建议使用第二类实现。

此外，关于LSTM, PaddlePaddle中还包含 :code:`paddle.networks.lstmemory_unit` 这一计算单元：

  * 不同于上述介绍的recurrent layer , :code:`paddle.networks.lstmemory_unit` 定义了LSTM单元在一个时间步内的计算过程，它并不是一个完整的recurrent layer，也不能接收序列数据作为输入；
  * :code:`paddle.networks.lstmemory_unit` 只能在recurrent_group中作为step function使用；

5. PaddlePaddle的softmax能否指定计算的维度
-----------------------------------------

PaddlePaddle的softmax不能指定计算维度，只能按行计算。
在图像任务中，对于NCHW，如果需要在C维度计算softmax，可以先使用 :code:`paddle.layer.switch_order` 改变维度顺序，即将NCHW转换成NHWC，再做一定的reshape，最后计算softmax。

6. PaddlePaddle是否支持维数可变的数据输入
------------------------------------------

PaddlePaddle提供的 :code:`paddle.data_type.dense_array` 支持维数可变的数据输入。在使用时，将对应数据层的维数设置成一个大于输入数据维数的值用于占位即可。
