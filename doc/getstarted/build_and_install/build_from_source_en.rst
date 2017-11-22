Build PaddlePaddle from Sources
==========================

.. _build_step:

How To Build
----------------

PaddlePaddle mainly uses `CMake <https://cmake.org>`_ and GCC, G++ as compile
tools. We recommend you to use our pre-built Docker image to run the build
to avoid installing dependencies by yourself. We have several build environment
Docker images `here <https://hub.docker.com/r/paddlepaddle/paddle_manylinux_devel/tags/>`_.
Then run:

.. code-block:: bash

   git clone https://github.com/PaddlePaddle/Paddle.git
   cd Paddle
   # run the following command if you are using docker
   docker run -it -v $PWD:/paddle -e "WITH_GPU=ON" -e "WITH_TESTING=OFF" paddlepaddle/paddle_manylinux_devel:cuda8.0_cudnn5 bash -x paddle/scripts/docker/build.sh
   # else run these commands
   mkdir build
   cd build
   cmake -DWITH_GPU=ON -DWITH_TESTING=OFF ..
   make

When the compile finishes, you can get the output whl package under
build/python/dist, then you can choose to install the whl on local
machine or copy it to the target machine.

.. code-block:: bash

   pip install python/dist/*.whl

.. _build_step:

Compile Dependencies
----------------

PaddlePaddle need the following dependencies when compiling, other dependencies
will be downloaded automatically.

.. csv-table:: PaddlePaddle Compile Dependencies
   :header: "Dependency", "Version", "Description"
   :widths: 10, 15, 30

   "CMake", ">=3.5", ""
   "GCC", "4.8.2", "Recommend devtools2 for CentOS"
   "Python", "2.7.x", "Need libpython2.7.so"
   "pip", ">=9.0", ""
   "numpy", "", ""
   "SWIG", ">=2.0", ""
   "Go", ">=1.8", "Optional"


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
    "WITH_DOUBLE", "Build with double precision", "OFF"
    "WITH_DSO", "Dynamically load CUDA libraries", "ON"
    "WITH_AVX", "Build with AVX support", "ON"
    "WITH_PYTHON", "Build with integrated Python interpreter", "ON"
    "WITH_STYLE_CHECK", "Check code style when building", "ON"
    "WITH_TESTING", "Build unit tests", "ON"
    "WITH_DOC", "Build documentaions", "OFF"
    "WITH_SWIG_PY", "Build Python SWIG interface for V2 API", "Auto"
    "WITH_GOLANG", "Build fault-tolerant parameter server written in go", "ON"

.. _build_options_blas:

BLAS/CUDA/Cudnn Options
--------------------------
BLAS
+++++

You can build PaddlePaddle with any of the below BLAS libraries:
`MKL <https://software.intel.com/en-us/intel-mkl>`_ ,
`ATLAS <http://math-atlas.sourceforge.net/>`_ ,
`OpenBlAS <http://www.openblas.net/>`_ and
`REFERENCE BLAS <http://www.netlib.org/blas/>`_ .

..  csv-table:: BLAS Options
    :header: "Option", "Description"
    :widths: 1, 7
    
    "MKL_ROOT", "${MKL_ROOT}/include must have mkl.h, ${MKL_ROOT}/lib must have mkl_core, mkl_sequential and mkl_intel_lp64 libs."
    "ATLAS_ROOT", "${ATLAS_ROOT}/include must have cblas.h，${ATLAS_ROOT}/lib must have cblas and atlas libs"
    "OPENBLAS_ROOT", "${OPENBLAS_ROOT}/include must have cblas.h，${OPENBLAS_ROOT}/lib must have OpenBlas libs."
    "REFERENCE_CBLAS_ROOT", "${REFERENCE_CBLAS_ROOT}/include must have cblas.h，${REFERENCE_CBLAS_ROOT}/lib must have cblas lib."

CUDA/Cudnn
+++++++++++

PaddlePaddle can build with any version later than Cudnn v2, and we intend to
keep on with latest cudnn versions. Be sure to run with the same version of cudnn
you built.

Pass Compile Options
++++++++++++++

You can pass compile options to use intended BLAS/CUDA/Cudnn libraries.
When running cmake command, it will search system paths like
:code:`/usr/lib\:/usr/local/lib` and then search paths that you
passed to cmake, i.e.

..  code-block:: bash

    cmake .. -DMKL_ROOT=/opt/mkl/ -DCUDNN_ROOT=/opt/cudnnv5

**NOTE: These options only take effect when running cmake for the first time, you need to clean the cmake cache or clean the build directory if you want to change it.**

