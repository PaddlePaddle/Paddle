从源码编译
======================

.. _build_step:

编译方法
----------------

PaddlePaddle主要使用 `CMake <https://cmake.org>`_ 以及GCC, G++作为编译工具。
我们推荐您使用PaddlePaddle Docker编译环境镜像完成编译，这样可以免去单独安装编译依赖的步骤，可选的不同编译环境Docker镜像
可以在 `这里 <https://hub.docker.com/r/paddlepaddle/paddle_manylinux_devel/tags/>`_ 找到。

如果您选择不使用Docker镜像，则需要在本机安装下面章节列出的 `编译依赖`_ 之后才能开始编译的步骤。

编译PaddlePaddle，需要执行：

.. code-block:: bash

   git clone https://github.com/PaddlePaddle/Paddle.git
   cd Paddle
   # 如果使用Docker编译环境，执行下面的命令编译CPU-Only的二进制
   docker run -it -v $PWD:/paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=OFF" paddlepaddle/paddle_manylinux_devel:cuda8.0_cudnn5 bash -x /paddle/paddle/scripts/docker/build.sh
   # 如果不使用Docker编译环境，执行下面的命令
   mkdir build
   cd build
   cmake -DWITH_GPU=OFF -DWITH_TESTING=OFF ..
   make

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

使用Docker的情况下，设置 :code:`RUN_TEST=ON` 和 :code:`WITH_TESTING=ON` 就会在完成编译之后，立即执行单元测试。
开启 :code:`WITH_GPU=ON` 可以指定同时执行GPU上的单元测试。

.. code-block:: bash

   docker run -it -v $PWD:/paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=ON" -e "RUN_TEST=ON" paddlepaddle/paddle_manylinux_devel:cuda8.0_cudnn5 bash -x /paddle/paddle/scripts/docker/build.sh

如果不使用Docker，可以执行ctest命令即可：

.. code-block:: bash

   mkdir build
   cd build
   cmake -DWITH_GPU=OFF -DWITH_TESTING=OFF ..
   make
   ctest
   # 指定执行其中一个单元测试 test_mul_op
   ctest -R test_mul_op

.. _compile_deps:

编译依赖
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

编译选项
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
    "WITH_TESTING", "是否开启单元测试", "ON"
    "WITH_DOC", "是否编译中英文文档", "OFF"
    "WITH_SWIG_PY", "是否编译PYTHON的SWIG接口，该接口可用于预测和定制化训练", "Auto"
    "WITH_GOLANG", "是否编译go语言的可容错parameter server", "ON"
    "WITH_MKL", "是否使用MKL数学库，如果为否则是用OpenBLAS", "ON"

BLAS
+++++

PaddlePaddle支持 `MKL <https://software.intel.com/en-us/intel-mkl>`_ 和
`OpenBlAS <http://www.openblas.net/>`_ 两种BLAS库。默认使用MKL。如果使用MKL并且机器含有AVX2指令集，
还会下载MKL-DNN数学库，详细参考 `这里 <https://github.com/PaddlePaddle/Paddle/tree/develop/doc/design/mkldnn#cmake>`_ 。

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
