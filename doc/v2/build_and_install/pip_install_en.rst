Install using pip
================================

You can use current widely used Python package management
tool `pip <https://pip.pypa.io/en/stable/installing/>`_
to install PaddlePaddle. This method can be used in
most of current Linux systems or MacOS.

.. _pip_install:

Install using pip
------------------------------

Run the following command to install PaddlePaddle on the current
machine, it will also download requirements.

  .. code-block:: bash

     pip install paddlepaddle

the default version is 0.12.0, cpu_avx_openblas, you can specify the versions to satisfy your demands, like:

  .. code-block:: bash

      pip install paddlepaddle==0.11.0

If you need to install a GPU-enabled version (cuda8.0_cudnn5_avx_openblas), you need to run:

  .. code-block:: bash

     pip install paddlepaddle-gpu

The default version is also 0.12.0, PaddlePaddle provides several versions of packages for different needs, as shown in the table:

=================================   ========================================
版本号                               版本说明
=================================   ========================================
paddlepaddle-gpu==0.12.0            0.12.0 built with CUDA 8.0 and cuDNN 5
paddlepaddle-gpu==0.11.0.post87     0.11.0 built with CUDA 8.0 and cuDNN 7
paddlepaddle-gpu==0.11.0.post8      0.11.0 built with CUDA 8.0 and cuDNN 5
paddlepaddle-gpu==0.11.0            0.11.0 built with CUDA 7.5 and cuDNN 5
=================================   ========================================

You can find all versions released of paddlepaddle-gpu in `Release History <https://pypi.org/project/paddlepaddle-gpu/#history>`_ .

If you wish to install the latest develop branch PaddlePaddle,
you can download the latest whl package from our CI system. Access
the below links, log in as guest, then click at the "Artifact"
tab, you'll find the download link of whl packages.

If the links below shows up the login form, just click "Log in as guest" to start the download:

.. image:: paddleci.png
   :scale: 50 %
   :align: center

..  csv-table:: whl package of each version
    :header: "version", "cp27-cp27mu", "cp27-cp27m"
    :widths: 1, 3, 3

    "cpu_avx_mkl", "`paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxCp27cp27mu/.lastSuccessful/paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl>`__", "`paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxCp27cp27mu/.lastSuccessful/paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl>`__"
    "cpu_avx_openblas", "`paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxOpenblas/.lastSuccessful/paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl>`__", "`paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxOpenblas/.lastSuccessful/paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl>`__"
    "cpu_noavx_openblas", "`paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuNoavxOpenblas/.lastSuccessful/paddlepaddle-latest-cp27-cp27mu-linux_x86_64.whl>`__", "`paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuNoavxOpenblas/.lastSuccessful/paddlepaddle-latest-cp27-cp27m-linux_x86_64.whl>`__"
    "cuda8.0_cudnn5_avx_mkl", "`paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda80cudnn5cp27cp27mu/.lastSuccessful/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl>`__", "`paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda80cudnn5cp27cp27mu/.lastSuccessful/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl>`__"
    "cuda8.0_cudnn7_avx_mkl", "`paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda8cudnn7cp27cp27mu/.lastSuccessful/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl>`__", "`paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda8cudnn7cp27cp27mu/.lastSuccessful/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl>`__"
    "cuda9.0_cudnn7_avx_mkl", "`paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda90cudnn7avxMkl/.lastSuccessful/paddlepaddle_gpu-latest-cp27-cp27mu-linux_x86_64.whl>`__", "`paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda90cudnn7avxMkl/.lastSuccessful/paddlepaddle_gpu-latest-cp27-cp27m-linux_x86_64.whl>`__"

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
