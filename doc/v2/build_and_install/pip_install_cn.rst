使用pip安装
================================

PaddlePaddle可以使用常用的Python包管理工具
`pip <https://pip.pypa.io/en/stable/installing/>`_
完成安装，并可以在大多数主流的Linux操作系统以及MacOS上执行。

.. _pip_install:

使用pip安装
------------------------------

执行下面的命令即可在当前机器上安装PaddlePaddle的运行时环境，并自动下载安装依赖软件。

  .. code-block:: bash

     pip install paddlepaddle

当前的默认版本为0.12.0，cpu_avx_openblas，您可以通过指定版本号来安装其它版本，例如:

  .. code-block:: bash

      pip install paddlepaddle==0.11.0


如果需要安装支持GPU的版本（cuda8.0_cudnn5_avx_openblas），需要执行：

  .. code-block:: bash

     pip install paddlepaddle-gpu

当前的默认版本也是0.12.0，PaddlePaddle针对不同需求提供了更多版本的安装包，部分列表如下：

=================================   ========================================
版本号                               版本说明
=================================   ========================================
paddlepaddle-gpu==0.12.0            使用CUDA 8.0和cuDNN 5编译的0.12.0版本
paddlepaddle-gpu==0.11.0.post87     使用CUDA 8.0和cuDNN 7编译的0.11.0版本
paddlepaddle-gpu==0.11.0.post8      使用CUDA 8.0和cuDNN 5编译的0.11.0版本
paddlepaddle-gpu==0.11.0            使用CUDA 7.5和cuDNN 5编译的0.11.0版本
=================================   ========================================

您可以在 `Release History <https://pypi.org/project/paddlepaddle-gpu/#history>`_ 中找到paddlepaddle-gpu的各个发行版本。

如果需要获取并安装最新的（开发分支）PaddlePaddle，可以从我们的CI系统中下载最新的whl安装包和c-api开发包并安装，
您可以从下面的表格中找到需要的版本：

如果在点击下面链接时出现如下登陆界面，点击“Log in as guest”即可开始下载：

.. image:: paddleci.png
   :scale: 50 %
   :align: center

..  csv-table:: 各个版本最新的whl包
    :header: "版本说明", "cp27-cp27mu", "cp27-cp27m"
    :widths: 1, 3, 3

    "cpu_avx_mkl", "`paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxCp27cp27mu/.lastSuccessful/paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl>`__", "`paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxCp27cp27mu/.lastSuccessful/paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl>`__"
    "cpu_avx_openblas", "`paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxOpenblas/.lastSuccessful/paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl>`__", "`paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxOpenblas/.lastSuccessful/paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl>`__"
    "cpu_noavx_openblas", "`paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuNoavxOpenblas/.lastSuccessful/paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl>`__", "`paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuNoavxOpenblas/.lastSuccessful/paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl>`_"
    "cuda8.0_cudnn5_avx_mkl", "`paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda80cudnn5cp27cp27mu/.lastSuccessful/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl>`__", "`paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda80cudnn5cp27cp27mu/.lastSuccessful/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl>`__"
    "cuda8.0_cudnn7_avx_mkl", "`paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda8cudnn7cp27cp27mu/.lastSuccessful/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl>`__", "`paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda8cudnn7cp27cp27mu/.lastSuccessful/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl>`__"
    "cuda9.0_cudnn7_avx_mkl", "`paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda90cudnn7avxMkl/.lastSuccessful/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl>`__", "`paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda90cudnn7avxMkl/.lastSuccessful/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl>`__"

.. _pip_dependency:

运行环境依赖
------------------------------

PaddlePaddle安装包由于不仅仅包含.py程序，而且包含了C++编写的部分，所以我们确保发布的二进制包可以支持主流的Linux操作系统，比如CentOS 6以上，Ubuntu 14.04以上，MacOS 10.12以上。

PaddlePaddle发布的安装包会尽量对齐 `manylinux1 <https://www.python.org/dev/peps/pep-0513/#the-manylinux1-policy>`_ 标准，通常使用CentOS 5作为编译环境。但由于CUDA库通常需要CentOS 6以上，而且CentOS 5即将停止维护，所以我们默认使用CentOS 6作为标准编译环境。

.. csv-table:: PaddlePaddle环境依赖
   :header: "依赖", "版本", "说明"
   :widths: 10, 15, 30

   "操作系统", "Linux, MacOS", "CentOS 6以上，Ubuntu 14.04以上，MacOS 10.12以上"
   "Python", "2.7.x", "暂时不支持Python3"
   "libc.so", "GLIBC_2.7", "glibc至少包含GLIBC_2.7以上的符号"
   "libstdc++.so", "GLIBCXX_3.4.11, CXXABI_1.3.3", "至少包含GLIBCXX_3.4.11, CXXABI_1.3.3以上的符号"
   "libgcc_s.so", "GCC_3.3", "至少包含GCC_3.3以上的符号"

.. _pip_faq:

安装常见问题和解决方法
------------------------------

- paddlepaddle*.whl is not a supported wheel on this platform.

  出现这个问题的主要原因是，没有找到和当前系统匹配的paddlepaddle安装包。请检查Python版本是否为2.7系列。另外最新的pip官方源中的安装包默认是manylinux1标准，需要使用最新的pip (>9.0.0) 才可以安装。可以使用下面的命令更新您的pip：

    .. code-block:: bash

       pip install --upgrade pip

  如果仍然存在问题，可以执行：

      .. code-block:: bash

         python -c "import pip; print(pip.pep425tags.get_supported())"

  获取当前系统支持的安装包格式，并检查和需安装的包是否匹配。pypi安装包可以在 `这个 <https://pypi.python.org/pypi/paddlepaddle/0.10.5>`_ 链接中找到。

  如果系统支持的是 linux_x86_64 而安装包是 manylinux1_x86_64 ，需要升级pip版本到最新； 如果系统支持 manylinux1_x86_64 而安装包（本地）是 linux_x86_64 ，可以重命名这个whl包为 manylinux1_x86_64 再安装。
