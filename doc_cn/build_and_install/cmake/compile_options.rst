PaddlePaddle的编译选项
=====================

PaddlePaddle的编译选项，包括生成CPU/GPU二进制文件、链接何种BLAS库等。用户可在调用cmake的时候设置它们，详细的cmake使用方法可以参考 `官方文档 <https://cmake.org/cmake-tutorial>`_ 。

Bool型的编译选项
--------------------
用户可在cmake的命令行中，通过使用 ``-D`` 命令设置该类编译选项，例如

..  code-block:: bash

    cmake .. -DWITH_GPU=OFF

..  csv-table:: Bool型的编译选项
    :widths: 1, 7, 2
    :file: compile_options.csv

路径相关的编译选项
--------------------
BLAS路径相关
+++++++++++++

PaddlePaddle支持以下任意一种BLAS库：`MKL <https://software.intel.com/en-us/intel-mkl>`_ ，`ATLAS <http://math-atlas.sourceforge.net/>`_ ，`OpenBlAS <http://www.openblas.net/>`_ 和 `REFERENCE BLAS <http://www.netlib.org/blas/>`_ 。

..  csv-table:: BLAS路径相关的编译选项
    :widths: 1, 2, 7
    :file: cblas_settings.csv

CUDA/Cudnn路径相关
++++++++++++++++++++

PaddlePaddle可以使用cudnn v2之后的任何一个版本来编译运行，但尽量请保持编译和运行使用的cudnn是同一个版本。 我们推荐使用最新版本的cudnn v5.1。

编译选项的设置
+++++++++++++

cmake编译时，首先在系统路径(/usr/lib\:/usr/local/lib)中搜索上述库，其次也会根据相关路径的编译选项来进行搜索。 有两种方式可以设置：

1. 使用 ``-D`` 命令指定，例如 

..  code-block:: bash

    cmake .. -DMKL_ROOT=/opt/mkl/ -DCUDNN_ROOT=/opt/cudnnv5

2. 在cmake命令前，通过环境变量指定，例如

..  code-block:: bash

    export MKL_ROOT=/opt/mkl
    export CUDNN_ROOT=/opt/cudnnv5
    cmake

注意：该类编译选项的设置，只在第一次cmake的时候有效。如果之后想要重新设置，推荐清理整个编译目录（``rm -rf``）后，再指定。
