Build from Sources
==========================

.. _build_step:

How To Build
----------------

PaddlePaddle mainly uses `CMake <https://cmake.org>`_ and GCC, G++ as compile
tools. We recommend you to use our pre-built Docker image to run the build
to avoid installing dependencies by yourself. We have several build environment
Docker images `here <https://hub.docker.com/r/paddlepaddle/paddle_manylinux_devel/tags/>`_ .

If you choose not to use Docker image for your build, you need to install the
below `Compile Dependencies`_ before run the build.

Then run:

.. code-block:: bash

   git clone https://github.com/PaddlePaddle/Paddle.git
   cd Paddle
   # run the following command to build a CPU-Only binaries if you are using docker
   docker run -it -v $PWD:/paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=OFF" paddlepaddle/paddle_manylinux_devel:cuda8.0_cudnn5 bash -x /paddle/paddle/scripts/docker/build.sh
   # else run these commands
   mkdir build
   cd build
   cmake -DWITH_GPU=OFF -DWITH_TESTING=OFF ..
   make

When the compile finishes, you can get the output whl package under
build/python/dist, then you can choose to install the whl on local
machine or copy it to the target machine.

.. code-block:: bash

   pip install build/python/dist/*.whl

If the machine has installed PaddlePaddle before, there are two methods:

.. code-block:: bash

   1. uninstall and reinstall
   pip uninstall paddlepaddle
   pip install build/python/dist/*.whl

   2. upgrade directly
   pip install build/python/dist/*.whl -U

.. _run_test:

Run Tests
----------------

If you wish to run the tests, you may follow the below steps:

When using Docker, set :code:`RUN_TEST=ON` and :code:`WITH_TESTING=ON` will run test immediately after the build.
Set :code:`WITH_GPU=ON` Can also run tests on GPU.

.. code-block:: bash

   docker run -it -v $PWD:/paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=ON" -e "RUN_TEST=ON" paddlepaddle/paddle_manylinux_devel:cuda8.0_cudnn5 bash -x paddle/paddle/scripts/docker/build.sh

If you don't use Docker, just run ctest will start the tests:

.. code-block:: bash

   mkdir build
   cd build
   cmake -DWITH_GPU=OFF -DWITH_TESTING=ON ..
   make
   ctest
   # run a single test like test_mul_op
   ctest -R test_mul_op


.. _compile_deps:

Compile Dependencies
----------------

PaddlePaddle need the following dependencies when compiling, other dependencies
will be downloaded automatically.

.. csv-table:: PaddlePaddle Compile Dependencies
   :header: "Dependency", "Version", "Description"
   :widths: 10, 15, 30

   "CMake", ">=3.2", ""
   "GCC", "4.8.2", "Recommend devtools2 for CentOS"
   "Python", "2.7.x", "Need libpython2.7.so"
   "pip", ">=9.0", ""
   "numpy", "", ""
   "SWIG", ">=2.0", ""
   "Go", ">=1.8", "Optional"


.. _build_options:

Build Options
----------------

Build options include whether build binaries for CPU or GPU, which BLAS
library to use etc. You may pass these settings when running cmake.
For detailed cmake tutorial please refer to `here <https://cmake.org/cmake-tutorial>`_ 。

.. _build_options_bool:

Bool Type Options
----------------

You can add :code:`-D` argument to pass such options, like:

..  code-block:: bash

    cmake .. -DWITH_GPU=OFF

..  csv-table:: Bool Type Options
    :header: "Option", "Description", "Default"
    :widths: 1, 7, 2

    "WITH_GPU", "Build with GPU support", "ON"
    "WITH_C_API", "Build only CAPI", "OFF"
    "WITH_DOUBLE", "Build with double precision", "OFF"
    "WITH_DSO", "Dynamically load CUDA libraries", "ON"
    "WITH_AVX", "Build with AVX support", "ON"
    "WITH_PYTHON", "Build with integrated Python interpreter", "ON"
    "WITH_STYLE_CHECK", "Check code style when building", "ON"
    "WITH_TESTING", "Build unit tests", "OFF"
    "WITH_DOC", "Build documentations", "OFF"
    "WITH_SWIG_PY", "Build Python SWIG interface for V2 API", "Auto"
    "WITH_GOLANG", "Build fault-tolerant parameter server written in go", "ON"
    "WITH_MKL", "Use MKL as BLAS library, else use OpenBLAS", "ON"


BLAS
+++++

PaddlePaddle supports `MKL <https://software.intel.com/en-us/intel-mkl>`_ and
`OpenBlAS <http://www.openblas.net/>`_ as BLAS library。By default it uses MKL.
If you are using MKL and your machine supports AVX2, MKL-DNN will also be downloaded
and used, for more `details <https://github.com/PaddlePaddle/Paddle/tree/develop/doc/design/mkldnn#cmake>`_ .

If you choose not to use MKL, then OpenBlAS will be used.

CUDA/cuDNN
+++++++++++

PaddlePaddle will automatically find CUDA and cuDNN when compiling and running.
parameter :code:`-DCUDA_ARCH_NAME=Auto` can be used to detect SM architecture
automatically in order to speed up the build.

PaddlePaddle can build with any version later than cuDNN v5.1, and we intend to
keep on with latest cuDNN versions. Be sure to run with the same version of cuDNN
you built.

Pass Compile Options
++++++++++++++

You can pass compile options to use intended BLAS/CUDA/Cudnn libraries.
When running cmake command, it will search system paths like
:code:`/usr/lib:/usr/local/lib` and then search paths that you
passed to cmake, i.e.

..  code-block:: bash

    cmake .. -DWITH_GPU=ON -DWITH_TESTING=OFF -DCUDNN_ROOT=/opt/cudnnv5

**NOTE: These options only take effect when running cmake for the first time, you need to clean the cmake cache or clean the build directory (** :code:`rm -rf` **) if you want to change it.**
