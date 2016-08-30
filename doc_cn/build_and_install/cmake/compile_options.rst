设置PaddlePaddle的编译选项
==========================

PaddlePaddle的编译选项可以在调用cmake的时候设置。cmake是一个跨平台的编译脚本，调用
cmake可以将cmake项目文件，生成各个平台的makefile。详细的cmake使用方法可以参考
`cmake的官方文档 <https://cmake.org/cmake-tutorial>`_ 。

PaddlePaddle的编译选项是可以控制PaddlePaddle生成CPU/GPU版本二进制，链接何种blas等等。所有的
编译选项列表如下

PaddlePaddle的编译选项
----------------------

bool型的编译选项
++++++++++++++++
设置下列编译选项时，可以在cmake的命令行设置。使用 -D命令即可。例如 
:code:`cmake -D WITH_GPU=OFF`

..  csv-table:: PaddlePaddle的bool型编译选项
    :widths: 1, 7, 2
    :file: compile_options.csv

blas相关的编译选项
++++++++++++++++++

PaddlePaddle可以使用 `MKL <https://software.intel.com/en-us/intel-mkl>`_ ，
`Atlas <http://math-atlas.sourceforge.net/>`_ ,
`OpenBlas <http://www.openblas.net/>`_ 和 
`refference Blas <http://www.netlib.org/blas/>`_ ，任意一种cblas实现。
通过编译时指定路径来实现引用各种blas。

cmake编译时会首先在系统路径(/usr/lib\:/usr/local/lib)中寻找这些blas的实现。同时
也会读取相关路径变量来进行搜索。路径变量为\:


..  csv-table:: PaddlePaddle的cblas编译选项
    :widths: 1, 9
    :header: "编译选项", "描述"
    :file: cblas_settings.csv

这些变量均可以使用 -D命令指定。例如 :code:`cmake -D MKL_ROOT=/opt/mkl/`。这些变
量也可以通过调用cmake命令前通过环境变量指定。例如

..  code-block:: bash

    export MKL_ROOT=/opt/mkl
    cmake

需要注意的是，这些变量只在第一次cmake的时候有效。如果在第一次cmake之后想要重新设
置这些变量，推荐清理( :code:`rm -rf` )掉编译目录后，再指定。

cuda/cudnn相关的编译选项
++++++++++++++++++++++++

PaddlePaddle可以使用 cudnn v2之后的任何一个cudnn版本来编译运行。但需要注意的是编译和
运行使用的cudnn尽量是同一个版本。推荐使用最新版本的cudnn v5.1。

在cmake配置时可以使用 :code:`CUDNN_ROOT` 来配置CUDNN的安装路径。使用的命令也是 
-D，例如 :code:`cmake -D CUDNN_ROOT=/opt/cudnnv5` 。

需要注意的是，这些变量只在第一次cmake的时候有效。如果在第一次cmake之后想要重新设
置这些变量，推荐清理( :code:`rm -rf` )掉编译目录后，再指定。
