#########
参数设置
#########

..  contents::

1. 如何选择SGD算法的学习率
--------------------------

在采用sgd/async_sgd进行训练时，一个重要的问题是选择正确的learning_rate。如果learning_rate太大，那么训练有可能不收敛，如果learning_rate太小，那么收敛可能很慢，导致训练时间过长。

通常做法是从一个比较大的learning_rate开始试，如果不收敛，那减少学习率10倍继续试验，直到训练收敛为止。那么如何判断训练不收敛呢？可以估计出如果模型采用不变的输出最小的cost0是多少。

如果训练过程的的cost明显高于这个常数输出的cost，那么我们可以判断为训练不收敛。举一个例子，假如我们是三分类问题，采用multi-class-cross-entropy作为cost，数据中0,1,2三类的比例为 :code:`0.2, 0.5, 0.3` , 那么常数输出所能达到的最小cost是 :code:`-(0.2*log(0.2)+0.5*log(0.5)+0.3*log(0.3))=1.03` 。如果训练一个pass（或者更早）后，cost还大于这个数，那么可以认为训练不收敛，应该降低学习率。

2. 如何设置学习率退火（learning rate annealing）
------------------------------------------------

在相应的优化算法里设置learning_rate_schedule及相关参数，以使用Adam算法为例，代码如下：

..  code-block:: python

    optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        learning_rate_decay_a=0.5,
        learning_rate_decay_b=0.75,
        learning_rate_schedule="poly",)

PaddlePaddle目前支持8种learning_rate_schedule，这8种learning_rate_schedule及其对应学习率计算方式如下：

* "constant"
  
  lr = learning_rate

* "poly"

  lr = learning_rate * pow(1 + learning_rate_decay_a * num_samples_processed, -learning_rate_decay_b)

  其中，num_samples_processed为已训练样本数，下同。

* "caffe_poly"

  lr = learning_rate * pow(1.0 - num_samples_processed / learning_rate_decay_a, learning_rate_decay_b)

* "exp"

  lr = learning_rate * pow(learning_rate_decay_a, num_samples_processed / learning_rate_decay_b)

* "discexp"

  lr = learning_rate * pow(learning_rate_decay_a, floor(num_samples_processed / learning_rate_decay_b))

* "linear"

  lr = max(learning_rate - learning_rate_decay_a * num_samples_processed, learning_rate_decay_b)

* "manual"

  这是一种按已训练样本数分段取值的学习率退火方法。使用该learning_rate_schedule时，用户通过参数 :code:`learning_rate_args` 设置学习率衰减因子分段函数，当前的学习率为所设置 :code:`learning_rate` 与当前的衰减因子的乘积。以使用Adam算法为例，代码如下：

  ..  code-block:: python

      optimizer = paddle.optimizer.Adam(
          learning_rate=1e-3,
          learning_rate_schedule="manual",
          learning_rate_args="1000:1.0,2000:0.9,3000:0.8",)

  在该示例中，当已训练样本数小于等于1000时，学习率为 :code:`1e-3 * 1.0`；当已训练样本数大于1000小于等于2000时，学习率为 :code:`1e-3 * 0.9`；当已训练样本数大于2000时，学习率为 :code:`1e-3 * 0.8`。

* "pass_manual"

  这是一种按已训练pass数分段取值的学习率退火方法。使用该learning_rate_schedule时，用户通过参数 :code:`learning_rate_args` 设置学习率衰减因子分段函数，当前的学习率为所设置 :code:`learning_rate` 与当前的衰减因子的乘积。以使用Adam算法为例，代码如下：

  ..  code-block:: python

      optimizer = paddle.optimizer.Adam(
          learning_rate=1e-3,
          learning_rate_schedule="pass_manual",
          learning_rate_args="1:1.0,2:0.9,3:0.8",)

  在该示例中，当已训练pass数小于等于1时，学习率为 :code:`1e-3 * 1.0`；当已训练pass数大于1小于等于2时，学习率为 :code:`1e-3 * 0.9`；当已训练pass数大于2时，学习率为 :code:`1e-3 * 0.8`。

3. 如何初始化参数
-----------------

默认情况下，PaddlePaddle使用均值0，标准差为 :math:`\frac{1}{\sqrt{d}}` 来初始化参数。其中 :math:`d` 为参数矩阵的宽度。这种初始化方式在一般情况下不会产生很差的结果。如果用户想要自定义初始化方式，PaddlePaddle目前提供两种参数初始化的方式\:

* 高斯分布。将 :code:`param_attr` 设置成 :code:`param_attr=ParamAttr(initial_mean=0.0, initial_std=1.0)`
* 均匀分布。将 :code:`param_attr` 设置成 :code:`param_attr=ParamAttr(initial_max=1.0, initial_min=-1.0)`

比如设置一个全连接层的参数初始化方式和bias初始化方式，可以使用如下代码。

..  code-block:: python

    hidden = fc_layer(input=ipt, param_attr=ParamAttr(initial_max=1.0, initial_min=-1.0),
                      bias_attr=ParamAttr(initial_mean=1.0, initial_std=0.0))

上述代码将bias全部初始化为1.0, 同时将参数初始化为 :code:`[1.0, -1.0]` 的均匀分布。

4. 如何共享参数
---------------

PaddlePaddle的参数使用名字 :code:`name` 作为参数的ID，相同名字的参数，会共享参数。设置参数的名字，可以使用 :code:`ParamAttr(name="YOUR_PARAM_NAME")` 来设置。更方便的设置方式，是使得要共享的参数使用同样的 :code:`ParamAttr` 对象。

简单的全连接网络，参数共享的配置示例为\:

..  literalinclude:: ../../python/paddle/trainer_config_helpers/tests/configs/shared_fc.py

这里 :code:`hidden_a` 和 :code:`hidden_b` 使用了同样的parameter和bias。并且softmax层的两个输入也使用了同样的参数 :code:`softmax_param`。

5. 如何加载预训练参数
------------------------

* 对加载预训练参数的层，设置其参数属性 :code:`is_static=True`，使该层的参数在训练过程中保持不变。以embedding层为例，代码如下：

..  code-block:: python

    emb_para = paddle.attr.Param(name='emb', is_static=True)
    paddle.layer.embedding(size=word_dim, input=x, param_attr=emb_para)


* 从模型文件将预训练参数载入 :code:`numpy.array`，在创建parameters后，使用 :code:`parameters.set()` 加载预训练参数。PaddlePaddle保存的模型参数文件前16字节为头信息，用户将参数载入 :code:`numpy.array` 时须从第17字节开始。以embedding层为例，代码如下：

..  code-block:: python

    def load_parameter(file_name, h, w):
        with open(file_name, 'rb') as f:
            f.read(16)  # skip header.
            return np.fromfile(f, dtype=np.float32).reshape(h, w)

    parameters = paddle.parameters.create(my_cost)
    parameters.set('emb', load_parameter(emb_param_file, 30000, 256))

6. 存储的参数格式是什么，如何和明文进行相互转化
--------------------------------------------------

PaddlePaddle保存的模型参数文件内容由16字节头信息和网络参数两部分组成。头信息中，1~4字节表示PaddlePaddle版本信息，请直接填充0；5~8字节表示每个参数占用的字节数，当保存的网络参数为float类型时为4，double类型时为8；9~16字节表示保存的参数总个数。

将PaddlePaddle保存的模型参数还原回明文时，可以使用相应数据类型的 :code:`numpy.array` 加载具体网络参数，此时可以跳过PaddlePaddle模型参数文件的头信息。若在PaddlePaddle编译时，未指定按照double精度编译，默认情况下按照float精度计算，保存的参数也是float类型。这时在使用 :code:`numpy.array` 时，一般设置 :code:`dtype=float32` 。示例如下：

..  code-block:: python

    def read_parameter(fname, width):
        s = open(fname).read()
        # skip header
        vec = np.fromstring(s[16:], dtype=np.float32)
        # width is the size of the corresponding layer
        np.savetxt(fname + ".csv", vec.reshape(width, -1),
                fmt="%.6f", delimiter=",")


将明文参数转化为PaddlePaddle可加载的模型参数时，首先构造头信息，再写入网络参数。下面的代码将随机生成的矩阵转化为可以被PaddlePaddle加载的模型参数。

..  code-block:: python

    def gen_rand_param(param_file, width, height, need_trans):
        np.random.seed()
        header = struct.pack("iil", 0, 4, height * width)
        param = np.float32(np.random.rand(height, width))
        with open(param_file, "w") as fparam:
            fparam.write(header + param.tostring())

7. A protocol message was rejected because it was too big
------------------------------------------------------------

如果在训练NLP相关模型时，出现以下错误：

..  code-block:: bash

    [libprotobuf ERROR google/protobuf/io/coded_stream.cc:171] A protocol message was rejected because it was too big (more than 67108864 bytes).  To increase the limit (or to disable these warnings), see CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.
    F1205 14:59:50.295174 14703 TrainerConfigHelper.cpp:59] Check failed: m->conf.ParseFromString(configProtoStr)

可能的原因是：传给dataprovider的某一个args过大，一般是由于直接传递大字典导致的。错误的define_py_data_sources2类似：

..  code-block:: python

     src_dict = dict()
     for line_count, line in enumerate(open(src_dict_path, "r")):
        src_dict[line.strip()] = line_count

     define_py_data_sources2(
        train_list,
        test_list,
        module="dataprovider",
        obj="process",
        args={"src_dict": src_dict})

解决方案是：将字典的地址作为args传给dataprovider，然后在dataprovider里面根据该地址加载字典。即define_py_data_sources2应改为：

..  code-block:: python

     define_py_data_sources2(
        train_list,
        test_list,
        module="dataprovider",
        obj="process",
        args={"src_dict_path": src_dict_path})

完整源码可参考 `sequence_recurrent <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/sequence_recurrent.py>`_ 示例。


