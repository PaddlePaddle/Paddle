.. _user_guide_use_recordio_as_train_data:

############################
使用RecordIO文件作为训练数据
############################

相比于 :ref:`user_guide_use_numpy_array_as_train_data`，
:ref:`user_guide_use_recordio_as_train_data` 的性能更好；
但是用户需要先将训练数据集转换成RecordIO文件格式，再使用
:code:`fluid.layers.open_files()` 层在神经网络配置中导入 RecordIO 文件。
用户还可以使用 :code:`fluid.layers.double_buffer()` 加速数据从内存到显存的拷贝，
使用 :code:`fluid.layers.Preprocessor` 工具进行数据增强。

将训练数据转换成RecordIO文件格式
################################

:code:`fluid.recordio_writer` 中，每个记录都是一个
:code:`vector<LoDTensor>`, 即一个支持序列信息的Tensor数组。这个数组包括训练所需
的所有特征。例如对于图像分类来说，这个数组可以包含图片和分类标签。

用户可以使用 :code:`fluid.recordio_writer.convert_reader_to_recordio_file()` 可以将
:ref:`user_guide_reader` 转换成一个RecordIO文件。或者可以使用
:code:`fluid.recordio_writer.convert_reader_to_recordio_files()` 将一个
:ref:`user_guide_reader` 转换成多个RecordIO文件。

具体使用方法为:

.. code-block:: python

   import paddle.fluid as fluid
   import numpy

   def reader_creator():
       def __impl__():
           for i in range(1000):
               yield [
                        numpy.random.random(size=[3,224,224], dtype="float32"),
                        numpy.random.random(size=[1], dtype="int64")
                     ]
       return __impl__

   img = fluid.layers.data(name="image", shape=[3, 224, 224])
   label = fluid.layers.data(name="label", shape=[1], dtype="int64")
   feeder = fluid.DataFeeder(feed_list=[img, label], place=fluid.CPUPlace())

   BATCH_SIZE = 32
   reader = paddle.batch(reader_creator(), batch_size=BATCH_SIZE)
   fluid.recordio_writer.convert_reader_to_recordio_file(
      "train.recordio", feeder=feeder, reader_creator=reader)

其中 :code:`reader_creator` 创建了一个 :code:`Reader`。
:ref:`_api_fluid_data_feeder_DataFeeder`
是将 :code:`Reader` 转换成 :code:`LoDTensor` 的工具。详细请参考
:ref:`user_guide_reader` 。

上述程序将 :code:`reader_creator` 的数据转换成了 :code:`train.recordio` 文件，
其中每一个record 含有 32 条样本。如果batch size会在训练过程中调整，
用户可以将每一个Record的样本数设置成1。并参考
:ref:`user_guide_use_recordio_as_train_data_use_op_create_batch`。


配置神经网络, 打开RecordIO文件
##############################

RecordIO文件转换好之后，用户可以使用 :code:`fluid.layers.open_files()`
打开文件，并使用 :code:`fluid.layers.read_file` 读取文件内容。
简单使用方法如下:

.. code-block:: python

   import paddle.fluid as fluid

   file_obj = fluid.layers.open_files(
     filenames=["train.recordio"],
     shape=[[3, 224, 224], [1]],
     lod_levels=[0, 0],
     dtypes=["float32", "int64"],
     pass_num=100
   )

   image, label = fluid.layers.read_file(file_obj)

其中如果设置了 :code:`pass_num` ，那么当所有数据读完后，会重新读取数据，
直到读取了 :code:`pass_num` 遍。



进阶使用
########


使用 :code:`fluid.layers.double_buffer()`
------------------------------------------

:code:`Double buffer` 使用双缓冲技术，将训练数据从内存中复制到显存中。配置双缓冲
需要使用 :code:`fluid.layers.double_buffer()` 修饰文件对象。 例如:

.. code-block:: python

   import paddle.fliud as fluid
   file_obj = fluid.layers.open_files(...)
   file_obj = fluid.layers.double_buffer(file_obj)

   image, label = fluid.layers.read_file(file_obj)

双缓冲技术可以参考
`Multiple buffering <https://en.wikipedia.org/wiki/Multiple_buffering>`_ 。

配置数据增强
------------

使用 :code:`fluid.layers.Preprocessor` 可以配置文件的数据增强方法。例如

.. code-block:: python

   import paddle.fluid as fluid
   file_obj = fluid.layers.open_files(...)
   preprocessor = fluid.layers.Preprocessor(reader=data_file)
   with preprocessor.block():
       image, label = preprocessor.inputs()
       image = image / 2
       label = label + 1
       preprocessor.outputs(image, label)

如上代码所示，使用 :code:`Preprocessor` 定义了一个数据增强模块，并在
:code:`with preprocessor.block()` 中定义了数据增强的具体操作。 用户通过配置
:code:`preprocessor.inputs()` 获得数据文件中的各个字段。 并用
:code:`preprocessor.outputs()` 标记预处理后的输出。

.. _user_guide_use_recordio_as_train_data_use_op_create_batch:

使用Op组batch
-------------

使用 :code:`fluid.layers.batch()` 可以在训练的过程中动态的组batch。例如

.. code-block:: python

   import paddle.fluid as fluid
   file_obj = fluid.layers.open_files(...)
   file_obj = fluid.layers.batch(file_obj, batch_size=32)

   img, label = fluid.layers.read_file(file_obj)

需要注意的是，如果数据集中的最后几个样本不能组成 :code:`batch_size` 大小的批量数据，
那么这几个样本直接组成一个批量数据进行训练。

读入数据的shuffle
-----------------

使用 :code:`fluid.layers.shuffle()` 可以在训练过程中动态重排训练数据。例如

.. code-block:: python

   import paddle.fluid as fluid
   file_obj = fluid.layers.open_files(...)
   file_obj = fliud.layers.shuffle(file_obj, buffer_size=8192)

   img, label = fliud.layers.read_file(file_obj)

需要注意的是:

1. :code:`shuffle` 实现方法是:
先读入 :code:`buffer_size` 条样本，再随机的选出样本进行训练。

2. :code:`shuffle` 中 :code:`buffer_size` 会占用训练内存，需要确定训练过程中内存
足够支持缓存 :code:`buffer_size` 条数据。
