使用deb包在Ubuntu上安装PaddlePaddle
===================================

PaddlePaddle目前支持使用deb包安装。Paddle的 :code:`deb` 安装包在ubuntu 14.04中正确，但理论上支持其他的 debian 发行版。


PaddlePaddle的ubuntu安装包分为四个版本，他们是 cpu、gpu、cpu-noavx、gpu-noavx 四个版本。其中 noavx 用于不支持AVX指令集的cpu。安装包的下载地址是\: https://github.com/baidu/Paddle/releases/


用户需要先将PaddlePaddle安装包下载到本地，然后执行如下 :code:`gdebi` 命令即可完成安装。

..  code-block:: shell

    gdebi paddle-*-cpu.deb

如果 :code:`gdebi` 没有安装,则需要使用 :code:`sudo apt-get install gdebi`, 来安装 :code:`gdebi` 。


或者使用下面一条命令安装.

..  code-block:: shell

    dpkg -i paddle-*-cpu.deb
    apt-get install -f

在 :code:`dpkg -i` 的时候如果报一些依赖未找到的错误是正常的，
在 :code:`apt-get install -f` 里会继续安装 PaddlePaddle。

需要注意的是，如果使用GPU版本的PaddlePaddle，请安装CUDA 7.5 和CUDNN 5到本地环境中，
并设置好对应的环境变量(LD_LIBRARY_PATH等等)。

安装完成后,可以使用命令 :code:`paddle version` 查看安装后的paddle 版本。可能的输出为

..  literalinclude:: paddle_version.txt

可能遇到的问题
--------------

libcudart.so/libcudnn.so找不到
++++++++++++++++++++++++++++++

安装完成PaddlePaddle后，运行 :code:`paddle train` 报错\:

..	code-block:: shell

	0831 12:36:04.151525  1085 hl_dso_loader.cc:70] Check failed: nullptr != *dso_handle For Gpu version of PaddlePaddle, it couldn't find CUDA library: libcudart.so Please make sure you already specify its path.Note: for training data on Cpu using Gpu version of PaddlePaddle,you must specify libcudart.so via LD_LIBRARY_PATH.

PaddlePaddle使用运行时动态连接CUDA的so，如果在 LD_LIBRARY_PATH里面找不到这些动态
库的话，会报寻找不到这些动态库。

解决方法很简单，就是将这些动态库加到环境变量里面。比较可能的命令如下。

..	code-block:: text

	export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

CUDA Driver找不到
+++++++++++++++++

运行 :code:`paddle train` 报错\:

..	code-block:: text

	F0831 12:39:16.699000  1090 hl_cuda_device.cc:530] Check failed: cudaSuccess == cudaStat (0 vs. 35) Cuda Error: CUDA driver version is insufficient for CUDA runtime version

PaddlePaddle运行时如果没有寻找到cuda的driver，变会报这个错误。解决办法是将cuda 
driver添加到LD_LIBRARY_PATH中。比较可能的命令如下。

..	code-block:: text

	export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

config文件找不到
++++++++++++++++

运行 :code:`paddle train` 得到结果\:

..	code-block:: text

	F0831 20:53:07.525789  1302 TrainerMain.cpp:94] Check failed: config != nullptr no valid config

PaddlePaddle在运行时找不到对应的config文件，说明命令行参数 :code:`config` 没有设置。
而这个一般说明PaddlePaddle已经安装完毕了。