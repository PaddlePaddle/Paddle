..  _user_guide_prepare_data:

########
准备数据
########

PaddlePaddle Fluid支持两种传入数据的方式:

1. 用户需要使用 :code:`fluid.layers.data`
配置数据输入层，并在 :code:`fluid.Executor` 或 :code:`fluid.ParallelExecutor`
中，使用 :code:`executor.run(feed=...)` 传入训练数据。

2. 用户需要先将训练数据
转换成 Paddle 识别的 :code:`fluid.recordio_writer` ， 再使用
:code:`fluid.layers.open_files` 以及 :code:`fluid.layers.reader` 配置数据读取。

这两种准备数据方法的比较如下:

.. _user_guide_prepare_data_comparision:

+------------+----------------------------------+---------------------------------------+
|            |        Feed数据                  |         使用Reader                    |
+============+==================================+=======================================+
| API接口    | :code:`executor.run(feed=...)`   |         :code:`fluid.layers.reader`       |
+------------+----------------------------------+---------------------------------------+
| 数据格式   |           Numpy Array            | :code:`fluid.recordio_writer` |
+------------+----------------------------------+---------------------------------------+
| 数据增强   | Python端使用其他库完成           | 使用Fluid中的Operator 完成            |
+------------+----------------------------------+---------------------------------------+
|   速度     |                 慢               |                 快                    |
+------------+----------------------------------+---------------------------------------+
| 推荐用途   |   调试模型                       |   工业训练                            |
+------------+----------------------------------+---------------------------------------+

这些准备数据的详细使用方法，请参考:

.. toctree::
   :maxdepth: 2

   feeding_data

Python Reader
#############

为了方便用户在Python中定义数据处理流程，PaddlePaddle Fluid支持 Python Reader，
具体请参考:

.. toctree::
   :maxdepth: 2

   reader.md
