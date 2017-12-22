Install Using pip
================================

You can use current widely used Python package management
tool `pip <https://pip.pypa.io/en/stable/installing/>`_
to install PaddlePaddle. This method can be used in
most of current Linux systems or MacOS.

.. _pip_install:

Install Using pip
------------------------------

Run the following command to install PaddlePaddle on the current
machine, it will also download requirements.

  .. code-block:: bash

     pip install paddlepaddle


If you wish to install GPU version, just run:

  .. code-block:: bash

     pip install paddlepaddle-gpu

If you wish to install the latest develop branch PaddlePaddle, 
you can download the latest whl package from our CI system. Access
the below links, log in as guest, then click at the "Artifact"
tab, you'll find the download link of whl packages.

If the links below shows up the login form, just click "Log in as guest" to start the download:

.. image:: paddleci.png
   :scale: 50 %
   :align: center

..  csv-table:: whl package of each version
    :header: "version", "cp27-cp27mu", "cp27-cp27m", "C-API"
    :widths: 1, 3, 3, 3

    "cpu_avx_mkl", "`paddlepaddle-0.11.0-cp27-cp27mu-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxCp27cp27mu/.lastSuccessful/paddlepaddle-0.11.0-cp27-cp27mu-linux_x86_64.whl>`_", "`paddlepaddle-0.11.0-cp27-cp27m-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxCp27cp27mu/.lastSuccessful/paddlepaddle-0.11.0-cp27-cp27m-linux_x86_64.whl>`_", "`paddle.tgz <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxCp27cp27mu/.lastSuccessful/paddle.tgz>`_"
    "cpu_avx_openblas", "`paddlepaddle-0.11.0-cp27-cp27mu-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxOpenblas/.lastSuccessful/paddlepaddle-0.11.0-cp27-cp27mu-linux_x86_64.whl>`_", "`paddlepaddle-0.11.0-cp27-cp27m-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxOpenblas/.lastSuccessful/paddlepaddle-0.11.0-cp27-cp27m-linux_x86_64.whl>`_", "Not Available"
    "cuda7.5_cudnn5_avx_mkl", "`paddlepaddle_gpu-0.11.0-cp27-cp27mu-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda75cudnn5cp27cp27mu/.lastSuccessful/paddlepaddle_gpu-0.11.0-cp27-cp27mu-linux_x86_64.whl>`_", "`paddlepaddle_gpu-0.11.0-cp27-cp27m-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda75cudnn5cp27cp27mu/.lastSuccessful/paddlepaddle_gpu-0.11.0-cp27-cp27m-linux_x86_64.whl>`_", "`paddle.tgz <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda75cudnn5cp27cp27mu/.lastSuccessful/paddle.tgz>`_"
    "cuda8.0_cudnn5_avx_mkl", "`paddlepaddle_gpu-0.11.0-cp27-cp27mu-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda80cudnn5cp27cp27mu/.lastSuccessful/paddlepaddle_gpu-0.11.0-cp27-cp27mu-linux_x86_64.whl>`_", "`paddlepaddle_gpu-0.11.0-cp27-cp27m-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda80cudnn5cp27cp27mu/.lastSuccessful/paddlepaddle_gpu-0.11.0-cp27-cp27m-linux_x86_64.whl>`_", "`paddle.tgz <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda80cudnn5cp27cp27mu/.lastSuccessful/paddle.tgz>`_"
    "cuda8.0_cudnn7_avx_mkl", "`paddlepaddle_gpu-0.11.0-cp27-cp27mu-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda8cudnn7cp27cp27mu/.lastSuccessful/paddlepaddle_gpu-0.11.0-cp27-cp27mu-linux_x86_64.whl>`_", "`paddlepaddle_gpu-0.11.0-cp27-cp27m-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda8cudnn7cp27cp27mu/.lastSuccessful/paddlepaddle_gpu-0.11.0-cp27-cp27m-linux_x86_64.whl>`_", "`paddle.tgz <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda8cudnn7cp27cp27mu/.lastSuccessful/paddle.tgz>`_"

.. _pip_dependency:

Runtime Dependency
------------------------------

PaddlePaddle installation packages (whl) does not only contain .py files,
but also binaries built from C++ code. We ensure that PaddlePaddle can
run on current mainline Linux distributions, like CentOS 6, Ubuntu 14.04
and MacOS 10.12.

PaddlePaddle whl packages are trying to satisfy
`manylinux1 <https://www.python.org/dev/peps/pep-0513/#the-manylinux1-policy>`_
standard, which uses CentOS 5 as default build environment. But CUDA libraries
seems only run on CentOS 6 at least, also, CentOS 5 is about to end its lifetime,
so we use CentOS 6 as default build environment.

.. csv-table:: PaddlePaddle Runtime Deps
   :header: "Dependency", "version", "description"
   :widths: 10, 15, 30

   "OS", "Linux, MacOS", "CentOS 6 or later，Ubuntu 14.04 or later，MacOS 10.12 or later"
   "Python", "2.7.x", "Currently Python3 is not supported"
   "libc.so", "GLIBC_2.7", "glibc at least include GLIBC_2.7 symbols"
   "libstdc++.so", "GLIBCXX_3.4.11, CXXABI_1.3.3", "At least include GLIBCXX_3.4.11, CXXABI_1.3.3 symbols"
   "libgcc_s.so", "GCC_3.3", "At least include GCC_3.3 symbols"

.. _pip_faq:

FAQ
------------------------------

- paddlepaddle*.whl is not a supported wheel on this platform.
  
  The main cause of this issue is that your current platform is
  not supported. Please check that you are using Python 2.7 series.
  Besides, pypi only supports manylinux1 standard, you'll need to
  upgrade your pip to >9.0.0. Then run the below command:

    .. code-block:: bash

       pip install --upgrade pip

  If the problem still exists, run the following command:

      .. code-block:: bash

         python -c "import pip; print(pip.pep425tags.get_supported())"

  Then you'll get supported package suffixes, then check if it matches
  the file name of the whl package. You can find default whl package at
  `here <https://pypi.python.org/pypi/paddlepaddle/0.10.5>`_

  If your system supports linux_x86_64 but the whl package is manylinux1_x86_64,
  you'll need to update pip to the latest version; If your system supports
  manylinux1_x86_64 but the whl package is linux_x86_64 you can rename the
  file to manylinux1_x86_64 suffix and then install.
