从源码编译PaddlePaddle
======================

.. _build_step:

编译方法
----------------

PaddlePaddle主要使用 `CMake <https://cmake.org>`_ 以及GCC, G++作为编译工具。
我们推荐您使用PaddlePaddle编译环境镜像完成编译，这样可以免去单独安装编译依赖的步骤，可选的不同编译环境
可以在 `这里 <https://hub.docker.com/r/paddlepaddle/paddle_manylinux_devel/tags/>`_ 找到。
编译PaddlePaddle，需要执行：

.. code-block:: bash

   git clone https://github.com/PaddlePaddle/Paddle.git
   cd Paddle
   # 如果使用Docker编译环境，执行下面的命令
   docker run -it -v $PWD:/paddle -e "WITH_GPU=ON" -e "WITH_TESTING=OFF" paddlepaddle/paddle_manylinux_devel:cuda8.0_cudnn5 bash -x paddle/scripts/docker/build.sh
   # 如果不使用Docker编译环境，执行下面的命令
   mkdir build
   cd build
   cmake -DWITH_GPU=ON -DWITH_TESTING=OFF ..
   make
   

编译完成后会在build/python/dist目录下生成输出的whl包，可以选在在当前机器安装也可以拷贝到目标机器安装：

.. code-block:: bash

   pip install python/dist/*.whl


.. _build_step:

编译依赖
----------------

PaddlePaddle编译需要使用到下面的依赖（包含但不限于），其他的依赖软件，会自动在编译时下载。

.. csv-table:: PaddlePaddle编译依赖
   :header: "依赖", "版本", "说明"
   :widths: 10, 15, 30

   "CMake", ">=3.5", ""
   "GCC", "4.8.2", "推荐使用CentOS的devtools2"
   "Python", "2.7.x", "依赖libpython2.7.so"
   "pip", ">=9.0", ""
   "numpy", "", ""
   "SWIG", ">=2.0", ""
   "Go", ">=1.8", "可选"


.. _build_options:

编译选项
----------------

PaddlePaddle的编译选项，包括生成CPU/GPU二进制文件、链接何种BLAS库等。用户可在调用cmake的时候设置它们，详细的cmake使用方法可以参考 `官方文档 <https://cmake.org/cmake-tutorial>`_ 。

.. _build_options_bool:

Bool型的编译选项
----------------

用户可在cmake的命令行中，通过使用 ``-D`` 命令设置该类编译选项，例如

..  code-block:: bash

    cmake .. -DWITH_GPU=OFF

..  csv-table:: Bool型的编译选项
    :header: "选项", "说明", "默认值"
    :widths: 1, 7, 2

    "WITH_GPU", "是否支持GPU。", "是"
    "WITH_DOUBLE", "是否使用双精度浮点数。", "否"
    "WITH_DSO", "是否运行时动态加载CUDA动态库，而非静态加载CUDA动态库。", "是"
    "WITH_AVX", "是否编译含有AVX指令集的PaddlePaddle二进制文件", "是"
    "WITH_PYTHON", "是否内嵌PYTHON解释器。", "是"
    "WITH_STYLE_CHECK", "是否编译时进行代码风格检查", "是"
    "WITH_TESTING", "是否开启单元测试", "是"
    "WITH_DOC", "是否编译中英文文档", "否"
    "WITH_SWIG_PY", "是否编译PYTHON的SWIG接口，该接口可用于预测和定制化训练", "自动"
    "WITH_GOLANG", "是否编译go语言的可容错parameter server", "是"

.. _build_options_blas:

BLAS/CUDA/Cudnn的编译选项
--------------------------
BLAS
+++++

PaddlePaddle支持以下任意一种BLAS库：`MKL <https://software.intel.com/en-us/intel-mkl>`_ ，`ATLAS <http://math-atlas.sourceforge.net/>`_ ，`OpenBlAS <http://www.openblas.net/>`_ 和 `REFERENCE BLAS <http://www.netlib.org/blas/>`_ 。

..  csv-table:: BLAS路径相关的编译选项
    :header: "编译选项", "描述", "注意"
    :widths: 1, 2, 7
    
    "MKL_ROOT", "${MKL_ROOT}/include下需要包含mkl.h，${MKL_ROOT}/lib目录下需要包含mkl_core，mkl_sequential和mkl_intel_lp64三个库。"
    "ATLAS_ROOT", "${ATLAS_ROOT}/include下需要包含cblas.h，${ATLAS_ROOT}/lib下需要包含cblas和atlas两个库。"
    "OPENBLAS_ROOT", "${OPENBLAS_ROOT}/include下需要包含cblas.h，${OPENBLAS_ROOT}/lib下需要包含openblas库。"
    "REFERENCE_CBLAS_ROOT", "${REFERENCE_CBLAS_ROOT}/include下需要包含cblas.h，${REFERENCE_CBLAS_ROOT}/lib下需要包含cblas库。"

CUDA/Cudnn
+++++++++++

PaddlePaddle可以使用cudnn v2之后的任何一个版本来编译运行，但尽量请保持编译和运行使用的cudnn是同一个版本。 我们推荐使用最新版本的cudnn v5.1。

编译选项的设置
++++++++++++++

PaddePaddle通过编译时指定路径来实现引用各种BLAS/CUDA/Cudnn库。cmake编译时，首先在系统路径(/usr/lib\:/usr/local/lib)中搜索这几个库，同时也会读取相关路径变量来进行搜索。 通过使用 ``-D`` 命令可以设置，例如 

..  code-block:: bash

    cmake .. -DMKL_ROOT=/opt/mkl/ -DCUDNN_ROOT=/opt/cudnnv5

注意：这几个编译选项的设置，只在第一次cmake的时候有效。如果之后想要重新设置，推荐清理整个编译目录（``rm -rf``）后，再指定。
