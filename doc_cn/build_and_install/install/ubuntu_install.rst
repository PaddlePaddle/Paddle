使用deb包在Ubuntu上安装PaddlePaddle
=============================

PaddlePaddle目前支持ubuntu 14.04版本使用deb包安装。更多的安装包PaddlePaddle会在近期提供。
欢迎大家贡献各个发行版的安装包(例如，ubuntu，centos，debian，gentoo)。

PaddlePaddle的ubuntu安装包分为两个版本，即CPU版本，和GPU版本，他们的下载地址是:

* CPU版本的PaddlePaddle安装包:  TBD
* GPU版本的PaddlePaddle安装包:  TBD

需要注意的是，目前PaddlePaddle的安装包只支持 
`AVX <https://en.wikipedia.org/wiki/Advanced_Vector_Extensions>`_
指令集的X86 CPU。如果系统使用不支持 `AVX`_ 指令集的CPU运行PaddlePaddle，那么需要从源码
编译PaddlePaddle，请参考 `编译文档 <../cmake/index.html>`_ 。

用户需要先将PaddlePaddle安装包下载到本地，然后执行如下命令即可完成安装。

..  code-block:: shell

    dpkg -i paddle-1.0.0-cpu.deb
    apt-get install -f

需要注意的是，如果使用GPU版本的PaddlePaddle，请安装CUDA 7.5 和CUDNN 5到本地环境中，并
设置好对应的环境变量(LD_LIBRARY_PATH等等)。
