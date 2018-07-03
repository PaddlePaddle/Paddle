.. _install_faq:

###############################
Compile, Install, and Unit Test
###############################

..  contents::

1. Insufficient CUDA driver version
----------------------------------------------------------------

Many users usually face issues like `Cuda Error: CUDA driver version is insufficient for CUDA runtime version` when running the PaddlePaddle GPU Docker image. The cause is that you may not map the local CUDA driver to a container directory.
You can solve the issue by running the following commands:

..  code-block:: bash

    $ export CUDA_SO="$(\ls usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
    $ export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
    $ docker run ${CUDA_SO} ${DEVICES} -it paddlepaddle/paddle:latest-gpu

For more infomation about Docker's installation and usage, please refer to `PaddlePaddle Docker documentation <http://www.paddlepaddle.org/docs/0.11.0/documentation/zh/getstarted/build_and_install/docker_install_en.html>`_ .


2. Version mismatch between PythonLibs and PythonInterpreter
----------------------------------------------------------------

It is a common bug when CMake looks up Python. If you install multiple versions of Python, Cmake may find the version mismatch between PythonLibs and PythonInterpreter . You are forced to specify a Python version, as follows.

    ..  code-block:: bash

        cmake .. -DPYTHON_EXECUTABLE=<exc_path> -DPYTHON_LIBRARY=<lib_path>  -DPYTHON_INCLUDE_DIR=<inc_path>

You should specify ``<exc_path>``, ``<lib_path>``, ``<inc_path>`` to your local paths.

3. PaddlePaddle version is 0.0.0
------------------------------------------------
This issue would happen when you run the code  `paddle version` or `cmake ..`

..  code-block:: bash

    CMake Warning at cmake/version.cmake:20 (message):
      Cannot add paddle version from git tag

You should pull all remote branches to your local machine with the command :code:`git fetch upstream` and then run :code:`cmake`

4. paddlepaddle\*.whl is not a supported wheel on this platform.
------------------------------------------------------------------------

The primary cause for this issue is that it can not find the correct PaddlePaddle installation package that matches your current system.The latest PaddlePaddle Python installation package supports Linux x86_64 and MacOS 10.12 os including Python2.7 and Pip 9.0.1.

You can upgrade Pip with the following command\:

..  code-block:: bash

    pip install --upgrade pip

If it does not work for you, you can run the command :code:`python -c "import pip; print(pip.pep425tags.get_supported())"` to get the suffix of Python package which your system may support and then compare it with the suffix of your installation.

If the system supports :code:`linux_x86_64` and  the installation package is :code:`manylinux1_x86_64`, you should upgrade pip to the latest 

if the system supports :code:`manylinux_x86_64` and the local installation package is :code:`linux1_x86_64`, you can rename the whl package to :code:`manylinux1_x86_64` and then try again.


5. ImportError: No module named v2
----------------------------------
Please uninstall Paddle V1 if you have installed it before.

..  code-block:: bash

    pip uninstall py_paddle paddle

Then install Python for PaddlePaddle , enter the build directory and run the following commands

pip install python/dist/paddle*.whl && pip install ../paddle/dist/py_paddle*.whl

6. Illegal instruction
-----------------------
This issue may be caused by the wrong usage of PaddlePaddle binary version which uses avx SIMD instructions to increase the performance of cpu. Please choose the correct version.

7.  Python unittest fails
--------------------------------

If the following python unittest testcases fail:

..  code-block:: bash

    24 - test_PyDataProvider (Failed)
    26 - test_RecurrentGradientMachine (Failed)
    27 - test_NetworkCompare (Failed)
    28 - test_PyDataProvider2 (Failed)
    32 - test_Prediction (Failed)
    33 - test_Compare (Failed)
    34 - test_Trainer (Failed)
    35 - test_TrainerOnePass (Failed)
    36 - test_CompareTwoNets (Failed)
    37 - test_CompareTwoOpts (Failed)
    38 - test_CompareSparse (Failed)
    39 - test_recurrent_machine_generation (Failed)
    40 - test_PyDataProviderWrapper (Failed)
    41 - test_config_parser (Failed)
    42 - test_swig_api (Failed)
    43 - layers_test (Failed)

Please check the PaddlePaddle unittest logs which may suggest the following:

..  code-block:: bash

    paddle package is already in your PYTHONPATH. But unittest need a clean environment.
    Please uninstall paddle package before start unittest. Try to 'pip uninstall paddle'.

The solution is:

* Remove old PaddlePaddle to make a clean environment for the unit tests. If PaddlePaddle package is already in Python's site-packages, unit tests would refer Python package in site-packages instead of Python package in the :code:`/python` directory of the source directory.  Setting :code:`PYTHONPATH` to :code:`/python` is also useless because Python's search path would give the priority to the installed Python package.


8. Failed to download the MKLML library
----------------------------------------------

..  code-block:: bash

    make[2]: *** [third_party/mklml/src/extern_mklml-stamp/extern_mklml-download] error 4
    make[1]: *** [CMakeFiles/extern_mklml.dir/all] error 2
    make[1]: *** waiting for the unfinished  jobs....

Cause: The network speed or SSL link causes the MKLML library to download unsuccessfully.

The solution is: manually download and install, the specific steps are as follows.

..  code-block:: bash

    // 1. enter the directory
    cd build/third_party/mklml/src/extern_mklml

    // 2. check the size of the package, normally 75M, if less than 75M, the download fails
    du -sh mklml_lnx_2018.0.1.20171007.tgz

    // 3. manually download and unzip and make the download success tag:
    wget --no-check-certificate https://github.com/01org/mkl-dnn/releases/download/v0.11/mklml_lnx_2018.0.1.20171007.tgz -c -O mklml_lnx_2018.0.1.20171007.tgz 
    tar zxf mklml_lnx_2018.0.1.20171007.tgz
    touch ../extern_mklml-stamp/extern_mklml-download

    // 4. then compile
    
