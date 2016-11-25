PaddlePaddle的命令行参数
========================

安装好PaddlePaddle后，在命令行直接敲击 ``paddle`` 或 ``paddle --help`` 会显示如下一些命令行参数。

* ``train`` Start a paddle_trainer
    启动一个PaddlePaddle训练进程。 ``paddle train`` 可以通过命令行参数 ``-local=true`` 启动一个单机的训练进程；也可以和 ``paddle pserver`` 一起使用启动多机的分布式训练进程。
* ``pserver`` Start a paddle_pserver_main
    在多机分布式训练下启动PaddlePaddle的parameter server进程。
* ``version`` Print paddle version
    用于打印当前PaddlePaddle的版本和编译选项相关信息。
* ``merge_model`` Start a paddle_merge_model
    用于将PaddlePaddle的模型参数文件和模型配置文件打包成一个文件，方便做部署分发。
* ``dump_config`` Dump the trainer config as proto string
    用于将PaddlePaddle的模型配置文件以proto string的格式打印出来。
* ``make_diagram``
    使用graphviz对PaddlePaddle的模型配置文件进行绘制。

更详细的介绍请参考各命令行参数文档。

..  toctree::
    :glob:

    paddle_train.rst
    paddle_pserver.rst
    paddle_version.rst
    merge_model.rst
    dump_config.rst
    make_diagram.rst
