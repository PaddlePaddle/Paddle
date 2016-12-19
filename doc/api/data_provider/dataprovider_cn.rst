.. _api_dataprovider:

DataProvider的介绍
==================

DataProvider是PaddlePaddle负责提供数据的模块。其作用是将数据传入内存或显存，让神经网络可以进行训练或预测。用户可以通过简单使用Python接口 :ref:`api_pydataprovider2` ，来自定义传数据的过程。如果有更复杂的使用，或者需要更高的效率，用户也可以在C++端自定义一个 ``DataProvider`` 。

PaddlePaddle需要用户在网络配置（trainer_config.py）中定义使用哪种DataProvider，并且在DataProvider中实现如何访问训练文件列表（train.list）或测试文件列表（test.list）。

- train.list和test.list存放在本地（推荐直接存放到训练目录，以相对路径引用)。一般情况下，两者均为纯文本文件，其中每一行对应一个数据文件地址：
  
  - 如果数据文件存于本地磁盘，这个地址则为它的绝对路径或相对路径(相对于PaddlePaddle程序运行时的路径)。
  - 地址也可以为hdfs文件路径，或者数据库连接路径等。
  - 由于这个地址会被DataProvider使用，因此，如何解析该地址也是用户自定义DataProvider时需要考虑的地方。
- 如果没有设置test.list，或设置为None，那么在训练过程中不会执行测试操作；否则，会根据命令行参数指定的测试方式，在训练过程中进行测试，从而防止过拟合。
