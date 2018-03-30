.. _install_faq:

###################
编译安装与单元测试
###################

..  contents::

1. 运行Docker GPU镜像出现 "CUDA driver version is insufficient"
----------------------------------------------------------------

用户在使用PaddlePaddle GPU的Docker镜像的时候，常常出现 `Cuda Error: CUDA driver version is insufficient for CUDA runtime version`, 原因在于没有把机器上CUDA相关的驱动和库映射到容器内部。
具体的解决方法是：

..  code-block:: bash

    $ export CUDA_SO="$(\ls usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
    $ export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
    $ docker run ${CUDA_SO} ${DEVICES} -it paddlepaddle/paddle:latest-gpu

更多关于Docker的安装与使用, 请参考 `PaddlePaddle Docker 文档 <http://www.paddlepaddle.org/docs/0.11.0/documentation/zh/getstarted/build_and_install/docker_install_cn.html>`_ 。


2. CMake源码编译, 找到的PythonLibs和PythonInterp版本不一致
----------------------------------------------------------------

这是目前CMake寻找Python的逻辑存在缺陷，如果系统安装了多个Python版本，CMake找到的Python库和Python解释器版本可能有不一致现象，导致编译PaddlePaddle失败。正确的解决方法是，
用户强制指定特定的Python版本，具体操作如下：

    ..  code-block:: bash

        cmake .. -DPYTHON_EXECUTABLE=<exc_path> -DPYTHON_LIBRARY=<lib_path>  -DPYTHON_INCLUDE_DIR=<inc_path>

用户需要指定本机上Python的路径：``<exc_path>``, ``<lib_path>``, ``<inc_path>``

3. CMake源码编译，Paddle版本号为0.0.0
--------------------------------------

如果运行 :code:`paddle version`, 出现 :code:`PaddlePaddle 0.0.0`；或者运行 :code:`cmake ..`，出现

..  code-block:: bash

    CMake Warning at cmake/version.cmake:20 (message):
      Cannot add paddle version from git tag

那么用户需要拉取所有的远程分支到本机，命令为 :code:`git fetch upstream`，然后重新cmake即可。

4. paddlepaddle\*.whl is not a supported wheel on this platform.
------------------------------------------------------------------------

出现这个问题的主要原因是，没有找到和当前系统匹配的paddlepaddle安装包。最新的paddlepaddle python安装包支持Linux x86_64和MacOS 10.12操作系统，并安装了python 2.7和pip 9.0.1。

更新 :code:`pip` 包的方法是\:

..  code-block:: bash

    pip install --upgrade pip

如果还不行，可以执行 :code:`python -c "import pip; print(pip.pep425tags.get_supported())"` 获取当前系统支持的python包的后缀，
并对比是否和正在安装的后缀一致。

如果系统支持的是 :code:`linux_x86_64` 而安装包是 :code:`manylinux1_x86_64` ，需要升级pip版本到最新；
如果系统支持 :code:`manylinux1_x86_64` 而安装包（本地）是 :code:`linux_x86_64` ，可以重命名这个whl包为 :code:`manylinux1_x86_64` 再安装。

5. 编译安装后执行 import paddle.v2 as paddle 报ImportError: No module named v2
------------------------------------------------------------------------------------------
先查看一下是否曾经安装过paddle v1版本，有的话需要先卸载：

pip uninstall py_paddle paddle

然后安装paddle的python环境, 在build目录下执行

pip install python/dist/paddle*.whl && pip install ../paddle/dist/py_paddle*.whl

6. 遇到“非法指令”或者是“illegal instruction”
--------------------------------------------

PaddlePaddle使用avx SIMD指令提高cpu执行效率，因此错误的使用二进制发行版可能会导致这种错误，请选择正确的版本。

7.  python相关的单元测试都过不了
--------------------------------

如果出现以下python相关的单元测试都过不了的情况：

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

并且查询PaddlePaddle单元测试的日志，提示：

..  code-block:: bash

    paddle package is already in your PYTHONPATH. But unittest need a clean environment.
    Please uninstall paddle package before start unittest. Try to 'pip uninstall paddle'.

解决办法是：

* 卸载PaddlePaddle包 :code:`pip uninstall paddle`, 清理掉老旧的PaddlePaddle安装包，使得单元测试有一个干净的环境。如果PaddlePaddle包已经在python的site-packages里面，单元测试会引用site-packages里面的python包，而不是源码目录里 :code:`/python` 目录下的python包。同时，即便设置 :code:`PYTHONPATH` 到 :code:`/python` 也没用，因为python的搜索路径是优先已经安装的python包。

8. 下载MKLML库失败
------------------

..  code-block:: bash

    make[2]: *** [third_party/mklml/src/extern_mklml-stamp/extern_mklml-download] 错误 4
    make[1]: *** [CMakeFiles/extern_mklml.dir/all] 错误 2
    make[1]: *** 正在等待未完成的任务....

原因：网速或SSL链接原因，导致MKLML库下载不成功。

解决办法是：手动下载并安装，具体步骤如下。

..  code-block:: bash

    // 1. 进入对应的目录
    cd build/third_party/mklml/src/extern_mklml

    // 2. 查看包的大小， 正常情况下是75M，如果小于75M，即下载失败：
    du -sh mklml_lnx_2018.0.1.20171007.tgz

    // 3. 手动下载且解压缩，并手动生成download成功标签：
    wget --no-check-certificate https://github.com/01org/mkl-dnn/releases/download/v0.11/mklml_lnx_2018.0.1.20171007.tgz -c -O mklml_lnx_2018.0.1.20171007.tgz 
    tar zxf mklml_lnx_2018.0.1.20171007.tgz
    touch ../extern_mklml-stamp/extern_mklml-download

    // 4. 接着编译即可

9. 在Mac上无法安装numpy等Python包，权限错误
------------------

Mac上对自带的Python和包有严格的权限保护，最好不要在自带的Python上安装。建议用virtualenv建立一个新的Python环境来操作。

virtualenv的基本原理是将机器上的Python运行所需的运行环境完整地拷贝一份。我们可以在一台机器上制造多份拷贝，并在这多个拷贝之间自由切换，这样就相当于在一台机器上拥有了多个相互隔离、互不干扰的Python环境。

下面简单介绍下如何用virtualenv为Paddle生成一个专用的Python环境：

安装virtualenv：
::::::::::::::::

virtualenv本身也是Python的一个包，可以用pip进行安装：

..  code-block:: bash

    sudo -H pip install virtualenv

由于virtualenv需要安装给系统自带的Python，因此需要使用sudo权限。

创建一个新的Python运行环境：
:::::::::::::::::::

..  code-block:: bash

    virtualenv --no-site-packages paddle

--no-site-packages 参数表示不拷贝已有的任何第三方包，创造一个完全干净的新Python环境。后面的paddle是我们为这个新创建的环境取的名字。

执行完这一步后，当前目录下应该会出现一个名为paddle（或者你取的其他名字）的目录。这个目录里保存了运行一个Python环境所需要的各种文件。

启动运行环境：
::::::::::::::::

..  code-block:: bash

    source paddle/bin/activate

执行后会发现命令提示符前面增加了(paddle)字样，说明已经成功启动了名为‘paddle’的Python环境。执行which python，可以发现使用的已经是刚刚创建的paddle目录下的Python。

在这个环境中，我们可以自由地进行Paddle的安装、使用和开发工作，无需担心对系统自带Python的影响。

退出运行环境：
:::::::::::::::

直接执行：

..  code-block:: bash

    deactivate

可以看到命令提示符前面的(paddle)字样消失。

自动启动某一Python环境：
::::::::::::::::

如果我们经常使用Paddle，我们每次打开终端后都需要执行一下source paddle/bin/activate来启动环境，比较繁琐。为了简便，可以修改终端的配置文件，来让终端每次启动后自动启动特定的Python环境。

执行:

..  code-block:: bash

    vi ~/.bash_profile

打开终端配置文件，并在文件的最后添加一行：

..  code-block:: bash

    source paddle/bin/activate

保存并关闭文件。

这样，每次打开终端时就会自动启动名为‘paddle’的Python环境了。
