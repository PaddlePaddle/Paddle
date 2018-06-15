从源码编译
======================

.. _requirements:

需要的软硬件
----------------

为了编译PaddlePaddle，我们需要

1. 一台电脑，可以装的是 Linux, Windows 或者 MacOS 操作系统
2. Docker

不需要依赖其他任何软件了。即便是 Python 和 GCC 都不需要，因为我们会把所有编译工具都安装进一个 Docker 镜像里。

.. _build_step:

编译方法
----------------

PaddlePaddle需要使用Docker环境完成编译，这样可以免去单独安装编译依赖的步骤，可选的不同编译环境Docker镜像
可以在 `这里 <https://hub.docker.com/r/paddlepaddle/paddle_manylinux_devel/tags/>`__ 找到，您也可以
在 `这里 <https://github.com/PaddlePaddle/Paddle/tree/develop/tools/manylinux1/>`__ 找到 paddle_manylinux_devel
镜像的编译以及使用方法。或者参考下述可选步骤，从源码中构建用于编译PaddlePaddle的Docker镜像。

如果您选择不使用Docker镜像，则需要在本机安装下面章节列出的 :ref:`编译依赖 <_compile_deps>` 之后才能开始编译的步骤。

编译PaddlePaddle，需要执行：

.. code-block:: bash

   # 1. 获取源码
   git clone https://github.com/PaddlePaddle/Paddle.git
   cd Paddle
   # 2. 可选步骤：源码中构建用于编译PaddlePaddle的Docker镜像
   docker build -t paddle:dev .
   # 3. 执行下面的命令编译CPU-Only的二进制
   docker run -it -v $PWD:/paddle -w /paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=OFF" paddlepaddle/paddle_manylinux_devel:cuda8.0_cudnn5 ./paddle/scripts/paddle_build.sh build
   # 4. 或者也可以使用为上述可选步骤构建的镜像（必须先执行第2步）
   docker run -it -v $PWD:/paddle -w /paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=OFF" paddle:dev ./paddle/scripts/paddle_build.sh build

注：上述命令把当前目录（源码树根目录）映射为 container 里的 :code:`/paddle` 目录。

编译完成后会在build/python/dist目录下生成输出的whl包，可以选在在当前机器安装也可以拷贝到目标机器安装：

.. code-block:: bash

   pip install build/python/dist/*.whl

如果机器中已经安装过PaddlePaddle，有两种方法：

.. code-block:: bash

   1. 先卸载之前的版本，再重新安装
   pip uninstall paddlepaddle
   pip install build/python/dist/*.whl

   2. 直接升级到更新的版本
   pip install build/python/dist/*.whl -U

.. _run_test:

执行单元测试
----------------

如果您期望在编译完成后立即执行所有的单元测试，可以按照下面的方法：

设置 :code:`RUN_TEST=ON` 和 :code:`WITH_TESTING=ON` 就会在完成编译之后，立即执行单元测试。
开启 :code:`WITH_GPU=ON` 可以指定同时执行GPU上的单元测试。

.. code-block:: bash

   docker run -it -v $PWD:/paddle -w /paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=ON" -e "RUN_TEST=ON" paddlepaddle/paddle_manylinux_devel:cuda8.0_cudnn5 ./paddle/scripts/paddle_build.sh test

如果期望执行其中一个单元测试，（比如 :code:`test_sum_op` ）：

.. code-block:: bash

   docker run -it -v $PWD:/paddle -w /paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=ON" -e "RUN_TEST=OFF" paddlepaddle/paddle_manylinux_devel:cuda8.0_cudnn5 /bin/bash
   ./paddle/scripts/paddle_build.sh build
   cd build
   ctest -R test_sum_op -V

.. _faq_docker:

常见问题
----------------

- 什么是 Docker?

  如果您没有听说 Docker，可以把它想象为一个类似 virtualenv 的系统，但是虚拟的不仅仅是 Python 的运行环境。

- Docker 还是虚拟机？

  有人用虚拟机来类比 Docker。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。

- 为什么用 Docker?

  把工具和配置都安装在一个 Docker image 里可以标准化编译环境。这样如果遇到问题，其他人可以复现问题以便帮助。

  另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。

- 我可以选择不用Docker吗？

  当然可以。大家可以用把开发工具安装进入 Docker image 一样的方式，把这些工具安装到本机。这篇文档介绍基于 Docker 的开发流程，是因为这个流程比其他方法都更简便。

- 学习 Docker 有多难？

  理解 Docker 并不难，大概花十分钟看一下 `如何使用Docker <https://zhuanlan.zhihu.com/p/19902938>`_ 。这可以帮您省掉花一小时安装和配置各种开发工具，以及切换机器时需要新安装的辛苦。别忘了 PaddlePaddle 更新可能导致需要新的开发工具。更别提简化问题复现带来的好处了。

- 我可以用 IDE 吗？

  当然可以，因为源码就在本机上。IDE 默认调用 make 之类的程序来编译源码，我们只需要配置 IDE 来调用 Docker 命令编译源码即可。

  很多 PaddlePaddle 开发者使用 Emacs。他们在自己的 `~/.emacs` 配置文件里加两行

  .. code-block:: emacs

    (global-set-key "\C-cc" 'compile)
    (setq compile-command "docker run --rm -it -v $(git rev-parse --show-toplevel):/paddle paddle:dev")

  就可以按 `Ctrl-C` 和 `c` 键来启动编译了。

- 可以并行编译吗？

  是的。我们的 Docker image 运行一个 `Paddle编译Bash脚本 <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/scripts/docker/build.sh>`_ 。这个脚本调用 `make -j$(nproc)` 来启动和 CPU 核一样多的进程来并行编译。

- Docker 需要 sudo

  如果用自己的电脑开发，自然也就有管理员权限（sudo）了。如果用公用的电脑开发，需要请管理员安装和配置好 Docker。此外，PaddlePaddle 项目在努力开始支持其他不需要 sudo 的集装箱技术，比如 rkt。

- 在 Windows/MacOS 上编译很慢

  Docker 在 Windows 和 MacOS 都可以运行。不过实际上是运行在一个 Linux 虚拟机上。可能需要注意给这个虚拟机多分配一些 CPU 和内存，以保证编译高效。具体做法请参考 `如何为Windows/Mac计算机上的Docker增加内存和虚拟机 <https://github.com/PaddlePaddle/Paddle/issues/627>`_ 。

- 磁盘不够

  本文中的例子里，`docker run` 命令里都用了 `--rm` 参数，这样保证运行结束之后的 containers 不会保留在磁盘上。可以用 `docker ps -a` 命令看到停止后但是没有删除的 containers。`docker build` 命令有时候会产生一些中间结果，是没有名字的 images，也会占用磁盘。可以参考 `如何删除Docker Container <https://zaiste.net/posts/removing_docker_containers/>`_ 来清理这些内容。


.. _compile_deps:

附录：编译依赖
----------------

PaddlePaddle编译需要使用到下面的依赖（包含但不限于），其他的依赖软件，会自动在编译时下载。

.. csv-table:: PaddlePaddle编译依赖
   :header: "依赖", "版本", "说明"
   :widths: 10, 15, 30

   "CMake", ">=3.2", ""
   "GCC", "4.8.2", "推荐使用CentOS的devtools2"
   "Python", "2.7.x", "依赖libpython2.7.so"
   "pip", ">=9.0", ""
   "numpy", "", ""
   "SWIG", ">=2.0", ""
   "Go", ">=1.8", "可选"


.. _build_options:

附录：编译选项
----------------

PaddlePaddle的编译选项，包括生成CPU/GPU二进制文件、链接何种BLAS库等。
用户可在调用cmake的时候设置它们，详细的cmake使用方法可以参考
`官方文档 <https://cmake.org/cmake-tutorial>`_ 。

在cmake的命令行中，通过使用 ``-D`` 命令设置该类编译选项，例如：

..  code-block:: bash

    cmake .. -DWITH_GPU=OFF

..  csv-table:: 编译选项说明
    :header: "选项", "说明", "默认值"
    :widths: 1, 7, 2

    "WITH_GPU", "是否支持GPU", "ON"
    "WITH_C_API", "是否仅编译CAPI", "OFF"
    "WITH_DOUBLE", "是否使用双精度浮点数", "OFF"
    "WITH_DSO", "是否运行时动态加载CUDA动态库，而非静态加载CUDA动态库。", "ON"
    "WITH_AVX", "是否编译含有AVX指令集的PaddlePaddle二进制文件", "ON"
    "WITH_PYTHON", "是否内嵌PYTHON解释器", "ON"
    "WITH_STYLE_CHECK", "是否编译时进行代码风格检查", "ON"
    "WITH_TESTING", "是否开启单元测试", "OFF"
    "WITH_DOC", "是否编译中英文文档", "OFF"
    "WITH_SWIG_PY", "是否编译PYTHON的SWIG接口，该接口可用于预测和定制化训练", "Auto"
    "WITH_GOLANG", "是否编译go语言的可容错parameter server", "OFF"
    "WITH_MKL", "是否使用MKL数学库，如果为否则是用OpenBLAS", "ON"

BLAS
+++++

PaddlePaddle支持 `MKL <https://software.intel.com/en-us/intel-mkl>`_ 和
`OpenBlAS <http://www.openblas.net/>`_ 两种BLAS库。默认使用MKL。如果使用MKL并且机器含有AVX2指令集，
还会下载MKL-DNN数学库，详细参考 `mkldnn设计文档 <https://github.com/PaddlePaddle/Paddle/tree/develop/doc/design/mkldnn#cmake>`_ 。

如果关闭MKL，则会使用OpenBLAS作为BLAS库。

CUDA/cuDNN
+++++++++++

PaddlePaddle在编译时/运行时会自动找到系统中安装的CUDA和cuDNN库进行编译和执行。
使用参数 :code:`-DCUDA_ARCH_NAME=Auto` 可以指定开启自动检测SM架构，加速编译。

PaddlePaddle可以使用cuDNN v5.1之后的任何一个版本来编译运行，但尽量请保持编译和运行使用的cuDNN是同一个版本。
我们推荐使用最新版本的cuDNN。

编译选项的设置
++++++++++++++

PaddePaddle通过编译时指定路径来实现引用各种BLAS/CUDA/cuDNN库。cmake编译时，首先在系统路径（ :code:`/usr/lib:/usr/local/lib` ）中搜索这几个库，同时也会读取相关路径变量来进行搜索。 通过使用 ``-D`` 命令可以设置，例如

..  code-block:: bash

    cmake .. -DWITH_GPU=ON -DWITH_TESTING=OFF -DCUDNN_ROOT=/opt/cudnnv5

**注意：这几个编译选项的设置，只在第一次cmake的时候有效。如果之后想要重新设置，推荐清理整个编译目录（** :code:`rm -rf` ）**后，再指定。**
