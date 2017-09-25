PaddlePaddle的编译选项
======================

PaddlePaddle的编译选项，包括生成CPU/GPU二进制文件、链接何种BLAS库等。用户可在调用cmake的时候设置它们，详细的cmake使用方法可以参考 `官方文档 <https://cmake.org/cmake-tutorial>`_ 。

Bool型的编译选项
----------------
用户可在cmake的命令行中，通过使用 ``-D`` 命令设置该类编译选项，例如

..  code-block:: bash

    cmake .. -DWITH_GPU=OFF

..  csv-table:: Bool型的编译选项
    :widths: 1, 7, 2
    :file: compile_options.csv

BLAS/CUDA/Cudnn的编译选项
--------------------------
BLAS
+++++

PaddlePaddle支持以下任意一种BLAS库：`MKL <https://software.intel.com/en-us/intel-mkl>`_ ，`ATLAS <http://math-atlas.sourceforge.net/>`_ ，`OpenBlAS <http://www.openblas.net/>`_ 和 `REFERENCE BLAS <http://www.netlib.org/blas/>`_ 。

..  csv-table:: BLAS路径相关的编译选项
    :widths: 1, 2, 7
    :file: cblas_settings.csv

CUDA/Cudnn
+++++++++++

PaddlePaddle可以使用cudnn v2之后的任何一个版本来编译运行，但尽量请保持编译和运行使用的cudnn是同一个版本。 我们推荐使用最新版本的cudnn v5.1。

编译选项的设置
++++++++++++++

PaddePaddle通过编译时指定路径来实现引用各种BLAS/CUDA/Cudnn库。cmake编译时，首先在系统路径(/usr/lib\:/usr/local/lib)中搜索这几个库，同时也会读取相关路径变量来进行搜索。 通过使用 ``-D`` 命令可以设置，例如 

..  code-block:: bash

    cmake .. -DMKL_ROOT=/opt/mkl/ -DCUDNN_ROOT=/opt/cudnnv5

注意：这几个编译选项的设置，只在第一次cmake的时候有效。如果之后想要重新设置，推荐清理整个编译目录（``rm -rf``）后，再指定。
