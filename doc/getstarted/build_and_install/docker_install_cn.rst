PaddlePaddle的Docker容器使用方式
================================

PaddlePaddle目前唯一官方支持的运行的方式是Docker容器。因为Docker能在所有主要操作系统（包括Linux，Mac OS X和Windows）上运行。 请注意，您需要更改 `Dockers设置 <https://github.com/PaddlePaddle/Paddle/issues/627>`_ 才能充分利用Mac OS X和Windows上的硬件资源。


PaddlePaddle发布的docker镜像使用说明
------------------------------

对于每一个PaddlePaddle版本，我们都会发布两种Docker镜像：开发镜像、运行镜像。运行镜像包括纯CPU版本和GPU版本以及其对应的非AVX版本。
我们会在 `dockerhub.com <https://hub.docker.com/r/paddledev/paddle/>`_ 提供最新的docker镜像，可以在"tags"标签下找到最新的Paddle镜像版本。
1. 开发镜像：:code:`paddlepaddle/paddle:<version>-dev`

    这个镜像包含了Paddle相关的开发工具以及编译和运行环境。用户可以使用开发镜像代替配置本地环境，完成开发，编译，发布，
    文档编写等工作。由于不同的Paddle的版本可能需要不同的依赖和工具，所以如果需要自行配置开发环境需要考虑版本的因素。
    开发镜像包含了以下工具：
    - gcc/clang
    - nvcc
    - Python
    - sphinx
    - woboq
    - sshd
    很多开发者会使用远程的安装有GPU的服务器工作，用户可以使用ssh登录到这台服务器上并执行 :code:`docker exec`进入开发镜像并开始工作，
    也可以在开发镜像中启动一个SSHD服务，方便开发者直接登录到镜像中进行开发:

    以交互容器方式运行开发镜像：

    .. code-block:: bash

        docker run -it --rm paddledev/paddle:<version>-dev /bin/bash

    或者，可以以后台进程方式运行容器：

    .. code-block:: bash

        docker run -d -p 2202:22 -p 8888:8888 paddledev/paddle:<version>-dev

    然后用密码 :code:`root` SSH进入容器：

    .. code-block:: bash

        ssh -p 2202 root@localhost

    SSH方式的一个优点是我们可以从多个终端进入容器。比如，一个终端运行vi，另一个终端运行Python。另一个好处是我们可以把PaddlePaddle容器运行在远程服务器上，并在笔记本上通过SSH与其连接。

2. 运行镜像：根据CPU、GPU和非AVX区分了如下4个镜像：
    - GPU/AVX：:code:`paddlepaddle/paddle:<version>-gpu`
    - GPU/no-AVX：:code:`paddlepaddle/paddle:<version>-gpu-noavx`
    - CPU/AVX：:code:`paddlepaddle/paddle:<version>`
    - CPU/no-AVX：:code:`paddlepaddle/paddle:<version>-noavx`

    纯CPU镜像以及GPU镜像都会用到AVX指令集，但是2008年之前生产的旧电脑不支持AVX。以下指令能检查Linux电脑是否支持AVX：

    .. code-block:: bash

       if cat /proc/cpuinfo | grep -i avx; then echo Yes; else echo No; fi

    如果输出是No，就需要选择使用no-AVX的镜像

    以上方法在GPU镜像里也能用，只是请不要忘记提前在物理机上安装GPU最新驱动。
    为了保证GPU驱动能够在镜像里面正常运行，我们推荐使用[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)来运行镜像。

    .. code-block:: bash

        nvidia-docker run -it --rm paddledev/paddle:0.10.0rc1-gpu /bin/bash

    注意: 如果使用nvidia-docker存在问题，你也许可以尝试更老的方法，具体如下，但是我们并不推荐这种方法。：

    .. code-block:: bash

        export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
        export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
        docker run ${CUDA_SO} ${DEVICES} -it paddledev/paddle:<version>-gpu

3. 使用运行镜像发布你的AI程序
    假设您已经完成了一个AI训练的python程序 :code:`a.py`，这个程序是您在开发机上使用开发镜像完成开发。此时您可以运行这个命令在开发机上进行测试运行：

    .. code-block:: bash

        docker run -it -v $PWD:/work paddle /work/a.py

    这里`a.py`包含的所有依赖假设都可以在Paddle的运行容器中。如果需要包含更多的依赖、或者需要发布您的应用的镜像，可以编写`Dockerfile`使用`FROM paddledev/paddle:<version>`
    创建和发布自己的AI程序镜像。

运行PaddlePaddle书籍
---------------------

Jupyter Notebook是一个开源的web程序，大家可以通过它制作和分享带有代码、公式、图表、文字的交互式文档。用户可以通过网页浏览文档。

PaddlePaddle书籍是为用户和开发者制作的一个交互式的Jupyter Nodebook。
如果您想要更深入了解deep learning，PaddlePaddle书籍一定是您最好的选择。

我们提供可以直接运行PaddlePaddle书籍的docker镜像，直接运行：

.. code-block:: bash

    docker run -p 8888:8888 paddlepaddle/book

然后在浏览器中输入以下网址：

.. code-block:: text

    http://localhost:8888/

就这么简单，享受您的旅程！

通过Docker容器开发PaddlePaddle
------------------------------

开发人员可以在Docker开发镜像中开发PaddlePaddle。这样开发人员可以以一致的方式在不同的平台上工作 - Linux，Mac OS X和Windows。

1. 构建开发镜像

   .. code-block:: bash

      git clone --recursive https://github.com/PaddlePaddle/Paddle
      cd Paddle
      docker build -t paddle:dev .


   请注意，默认情况下，:code:`docker build` 不会将源码导入到镜像中并编译它。如果我们想这样做，需要构建完开发镜像，然后执行：

   .. code-block:: bash

      docker run -v $PWD:/paddle -e "WITH_GPU=OFF" -e "WITH_AVX=ON" -e "TEST=OFF" paddle:dev


2. 运行开发环境

   当我们编译好了 :code:`paddle:dev`， 我们可以在docker容器里做开发，源代码可以通过挂载本地文件来被载入Docker的开发环境里面：

   .. code-block:: bash

      docker run -d -p 2202:22 -v $PWD:/paddle paddle:dev sshd

   以上代码会启动一个带有PaddlePaddle开发环境的docker容器，源代码会被挂载到 :code:`/paddle` 。

   以上的 :code:`docker run` 命令其实会启动一个在2202端口监听的SSHD服务器。这样，我们就能SSH进入我们的开发容器了：

   .. code-block:: bash

      ssh root@localhost -p 2202

3. 在Docker开发环境中编译与安装PaddlPaddle代码

   当在容器里面的时候，可以用脚本 :code:`paddle/scripts/docker/build.sh` 来编译、安装与测试PaddlePaddle：

   .. code-block:: bash

      /paddle/paddle/scripts/docker/build.sh

   以上指令会在 :code:`/paddle/build` 中编译PaddlePaddle。通过以下指令可以运行单元测试：

   .. code-block:: bash

      cd /paddle/build
      ctest


文档
----

Paddle的Docker开发镜像带有一个通过 `woboq code browser
<https://github.com/woboq/woboq_codebrowser>`_ 生成的HTML版本的C++源代码，便于用户浏览C++源码。

只要在Docker里启动PaddlePaddle的时候给它一个名字，就可以再运行另一个Nginx Docker镜像来服务HTML代码：

.. code-block:: bash

   docker run -d --name paddle-cpu-doc paddle:<version>-dev
   docker run -d --volumes-from paddle-cpu-doc -p 8088:80 nginx

接着我们就能够打开浏览器在 http://localhost:8088/paddle/ 浏览代码。
