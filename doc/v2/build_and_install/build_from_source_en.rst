Build from Sources
==========================

.. _requirements:

Requirements
----------------

To build PaddlePaddle, you need

1. A computer -- Linux, Windows, MacOS.
2. Docker.

Nothing else.  Not even Python and GCC, because you can install all build tools into a Docker image.
We run all the tools by running this image.

.. _build_step:

How To Build
----------------

You need to use Docker to build PaddlePaddle
to avoid installing dependencies by yourself. We have several pre-built
Docker images `here <https://hub.docker.com/r/paddlepaddle/paddle_manylinux_devel/tags/>`_ ,
you can also find how to build and use paddle_manylinux_devel Docker image from
`here <https://github.com/PaddlePaddle/Paddle/tree/develop/tools/manylinux1/>`__
Or you can build your own image from source as the optional step below:

If you don't wish to use docker，you need to install several compile dependencies manually as :ref:`Compile Dependencies <_compile_deps>` shows to start compilation.

.. code-block:: bash

   # 1. clone the source code
   git clone https://github.com/PaddlePaddle/Paddle.git
   cd Paddle
   # 2. Optional: build development docker image from source
   docker build -t paddle:dev .
   # 3. Run the following command to build a CPU-Only binaries
   docker run -it -v $PWD:/paddle -w /paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=OFF" paddlepaddle/paddle_manylinux_devel:cuda8.0_cudnn5 ./paddle/scripts/paddle_build.sh build
   # 4. Or, use your built Docker image to build PaddlePaddle (must run step 2)
   docker run -it -v $PWD:/paddle -w /paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=OFF" paddle:dev ./paddle/scripts/paddle_build.sh build

NOTE: The above command try to mount the current working directory (root directory of source code)
into :code:`/paddle` directory inside docker container.

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

   docker run -it -v $PWD:/paddle -w /paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=ON" -e "RUN_TEST=ON" paddlepaddle/paddle_manylinux_devel:cuda8.0_cudnn5 ./paddle/scripts/paddle_build.sh test

If you wish to run only one unit test, like :code:`test_sum_op`:

.. code-block:: bash

   docker run -it -v $PWD:/paddle -w /paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=ON" -e "RUN_TEST=OFF" paddlepaddle/paddle_manylinux_devel:cuda8.0_cudnn5 /bin/bash
   ./paddle/scripts/paddle_build.sh build
   cd build
   ctest -R test_sum_op -V

.. _faq_docker:

Frequently Asked Questions
---------------------------

- What is Docker?

  If you haven't heard of it, consider it something like Python's virtualenv.

- Docker or virtual machine?

  Some people compare Docker with VMs, but Docker doesn't virtualize any hardware nor running a guest OS, which means there is no compromise on the performance.

- Why Docker?

  Using a Docker image of build tools standardizes the building environment, which makes it easier for others to reproduce your problems and to help.

  Also, some build tools don't run on Windows or Mac or BSD, but Docker runs almost everywhere, so developers can use whatever computer they want.

- Can I choose not to use Docker?

  Sure, you don't have to install build tools into a Docker image; instead, you can install them on your local computer.  This document exists because Docker would make the development way easier.

- How difficult is it to learn Docker?

    It takes you ten minutes to read `an introductory article <https://docs.docker.com/get-started>`_ and saves you more than one hour to install all required build tools, configure them, especially when new versions of PaddlePaddle require some new tools.  Not even to mention the time saved when other people trying to reproduce the issue you have.

- Can I use my favorite IDE?

  Yes, of course.  The source code resides on your local computer, and you can edit it using whatever editor you like.

  Many PaddlePaddle developers are using Emacs.  They add the following few lines into their `~/.emacs` configure file:

  .. code-block:: emacs

    (global-set-key "\C-cc" 'compile)
    (setq compile-command "docker run --rm -it -v $(git rev-parse --show-toplevel):/paddle paddle:dev")

  so they could type `Ctrl-C` and `c` to build PaddlePaddle from source.

- Does Docker do parallel building?

  Our building Docker image runs a  `Bash script <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/scripts/docker/build.sh>`_ , which calls `make -j$(nproc)` to starts as many processes as the number of your CPU cores.

- Docker requires sudo

  An owner of a computer has the administrative privilege, a.k.a., sudo, and Docker requires this privilege to work properly.  If you use a shared computer for development, please ask the administrator to install and configure Docker.  We will do our best to support rkt, another container technology that doesn't require sudo.

- Docker on Windows/MacOS builds slowly

  On Windows and MacOS, Docker containers run in a Linux VM.  You might want to give this VM some more memory and CPUs so to make the building efficient.  Please refer to `this issue  <https://github.com/PaddlePaddle/Paddle/issues/627>`_ for details.

- Not enough disk space

  Examples in this article use option `--rm` with the `docker run` command.  This option ensures that stopped containers do not exist on hard disks.  We can use `docker ps -a` to list all containers, including stopped.  Sometimes `docker build` generates some intermediate dangling images, which also take disk space.  To clean them, please refer to `this article <https://zaiste.net/posts/removing_docker_containers/>`_ .

.. _compile_deps:

Appendix: Compile Dependencies
-------------------------------

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

Appendix: Build Options
-------------------------

Build options include whether build binaries for CPU or GPU, which BLAS
library to use etc. You may pass these settings when running cmake.
For detailed cmake tutorial please refer to `here <https://cmake.org/cmake-tutorial>`__ 。


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
    "WITH_GOLANG", "Build fault-tolerant parameter server written in go", "OFF"
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
++++++++++++++++++++++

You can pass compile options to use intended BLAS/CUDA/Cudnn libraries.
When running cmake command, it will search system paths like
:code:`/usr/lib:/usr/local/lib` and then search paths that you
passed to cmake, i.e.

..  code-block:: bash

    cmake .. -DWITH_GPU=ON -DWITH_TESTING=OFF -DCUDNN_ROOT=/opt/cudnnv5

**NOTE: These options only take effect when running cmake for the first time, you need to clean the cmake cache or clean the build directory (** :code:`rm -rf` **) if you want to change it.**
