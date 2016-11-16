命令行参数
==========

安装好的PaddlePaddle脚本包括多条命令，他们是

* paddle train即为PaddlePaddle的训练进程。可以使用paddle train完成单机多显卡多线程的训
  练。也可以和paddle pserver组合使用，完成多机训练。
* paddle pserver为PaddlePaddle的parameter server进程。负责多机训练中的参数聚合工作。
* paddle version可以打印出PaddlePaddle的版本和编译时信息。
* merge_model 可以将PaddlePaddle的模型和配置打包成一个文件。方便部署分发。
* dump_config 可以将PaddlePaddle的训练模型以proto string的格式打印出来
* make_diagram 可以使用graphviz对PaddlePaddle的网络模型进行绘制，方便调试使用。

更详细的介绍请参考各个命令的命令行参数文档。

..  toctree::
    :glob:

    paddle_train.rst
    paddle_pserver.rst
    paddle_version.rst
    merge_model.rst
    dump_config.rst
    make_diagram.rst
