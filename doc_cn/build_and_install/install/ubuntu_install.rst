Ubuntu部署PaddlePaddle
===================================

PaddlePaddle提供了deb安装包，并在ubuntu 14.04做了完备测试，理论上也支持其他的debian发行版。

安装
------

安装包的下载地址是\: https://github.com/PaddlePaddle/Paddle/releases

它包含四个版本\:

* cpu版本: 支持主流intel x86处理器平台, 支持avx指令集。

* cpu-noavx版本：支持主流intel x86处理器平台，不支持avx指令集。

* gpu版本：支持主流intel x86处理器平台，支持nvidia cuda平台，支持avx指令集。

* gpu-noavx版本：支持主流intel x86处理器平台，支持nvidia cuda平台，不支持avx指令级。

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

..  literalinclude:: paddle_version.txt

可能遇到的问题
--------------

如何设置gpu版本运行时cuda环境运行GPU版本
++++++++++++++++++++++++++++++++++++++++

如果使用GPU版本的PaddlePaddle，请安装CUDA 7.5 和CUDNN 5到本地环境中，并设置：

.. code-block:: shell
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH
  export PATH=/usr/local/cuda/bin:$PATH


libcudart.so/libcudnn.so找不到
++++++++++++++++++++++++++++++

安装完成后，运行 :code:`paddle train` 报错\:

..	code-block:: shell

	0831 12:36:04.151525  1085 hl_dso_loader.cc:70] Check failed: nullptr != *dso_handle For Gpu version of PaddlePaddle, it couldn't find CUDA library: libcudart.so Please make sure you already specify its path.Note: for training data on Cpu using Gpu version of PaddlePaddle,you must specify libcudart.so via LD_LIBRARY_PATH.

原因是未设置cuda运行时环境变量，请参考** 设置gpu版本运行时cuda环境** 解决方案。

