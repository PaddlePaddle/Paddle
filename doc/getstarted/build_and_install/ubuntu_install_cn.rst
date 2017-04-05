Ubuntu部署PaddlePaddle
===================================

PaddlePaddle提供了ubuntu 14.04 deb安装包。

安装
------

安装包的下载地址是\: https://github.com/PaddlePaddle/Paddle/releases

它包含四个版本\:

* cpu版本: 支持主流x86处理器平台, 使用了avx指令集。

* cpu-noavx版本：支持主流x86处理器平台，没有使用avx指令集。

* gpu版本：支持主流x86处理器平台，支持nvidia cuda平台，使用了avx指令集。

* gpu-noavx版本：支持主流x86处理器平台，支持nvidia cuda平台，没有使用avx指令集。

下载完相关安装包后，执行:

..  code-block:: shell

    sudo apt-get install gdebi
    gdebi paddle-*-cpu.deb

或者:

..  code-block:: shell

    dpkg -i paddle-*-cpu.deb
    apt-get install -f


在 :code:`dpkg -i` 的时候如果报一些依赖未找到的错误是正常的，
在 :code:`apt-get install -f` 里会继续安装 PaddlePaddle。

安装完成后，可以使用命令 :code:`paddle version` 查看安装后的paddle 版本:

..  code-block:: shell

    PaddlePaddle 0.8.0b1, compiled with
        with_avx: ON
        with_gpu: OFF
        with_double: OFF
        with_python: ON
        with_rdma: OFF
        with_timer: OFF
        with_predict_sdk:


可能遇到的问题
--------------

libcudart.so/libcudnn.so找不到
++++++++++++++++++++++++++++++

安装完成后，运行 :code:`paddle train` 报错\:

..  code-block:: shell

      0831 12:36:04.151525  1085 hl_dso_loader.cc:70] Check failed: nullptr != *dso_handle For Gpu version of PaddlePaddle, it couldn't find CUDA library: libcudart.so Please make sure you already specify its path.Note: for training data on Cpu using Gpu version of PaddlePaddle,you must specify libcudart.so via LD_LIBRARY_PATH.

原因是未设置cuda运行时环境变量。 如果使用GPU版本的PaddlePaddle，请安装CUDA 7.5 和CUDNN 5到本地环境中，并设置：

..  code-block:: shell

    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda/bin:$PATH

