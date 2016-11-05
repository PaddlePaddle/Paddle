PaddlePaddle的数据提供(DataProvider)介绍
========================================

数据提供(DataProvider)是PaddlePaddle负责提供数据的模块。其作用是将训练数据传入内存或者显存，让神经网络可以进行训练。简单的使用，用户可以使用Python的 :code:`PyDataProvider` 来自定义传数据的过程。如果有更复杂的使用，或者需要更高的效率，用户也可以在C++端自定义一个 :code:`DataProvider` 。

PaddlePaddle需要用户在网络配置(trainer_config.py)中定义使用哪种DataProvider及其参数，训练文件列表(train.list)和测试文件列表(test.list)。

其中，train.list和test.list均为本地的两个文件(推荐直接放置到训练目录，以相对路径引用)。如果test.list不设置，或者设置为None，那么在训练过程中，不会执行测试操作。否则，会根据命令行参数指定的测试方式，在训练过程中进行测试，从而防止过拟合。

一般情况下，train.list和test.list为纯文本文件，一行对应一个数据文件，数据文件存放在本地磁盘中。将文件的绝对路径或相对路径(相对于PaddlePaddle程序运行时的路径)写在train.list和test.list中。当然，train.list和test.list也可以放置hdfs文件路径，或者数据库连接地址等等。

用户在DataProvider中需要实现如何访问其中每一个文件。DataProvider的具体用法和如何实现一个新的DataProvider，请参考下述文章:

..	toctree::

	pydataprovider2.rst
	write_new_dataprovider.rst
