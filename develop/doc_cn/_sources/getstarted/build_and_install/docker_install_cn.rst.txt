PaddlePaddle的Docker容器使用方式
================================

PaddlePaddle目前唯一官方支持的运行的方式是Docker容器。因为Docker能在所有主要操作系统（包括Linux，Mac OS X和Windows）上运行。 请注意，您需要更改 `Dockers设置 <https://github.com/PaddlePaddle/Paddle/issues/627>`_ 才能充分利用Mac OS X和Windows上的硬件资源。


纯CPU和GPU的docker镜像使用说明
------------------------------

对于每一个PaddlePaddle版本，我们都会发布两个Docker镜像：纯CPU的和GPU的。
我们通过设置 `dockerhub.com <https://hub.docker.com/r/paddledev/paddle/>`_ 自动生成最新的docker镜像：
`paddledev/paddle:0.10.0rc1-cpu` 和 `paddledev/paddle:0.10.0rc1-gpu`。

以交互容器方式运行纯CPU的镜像：

.. code-block:: bash

    docker run -it --rm paddledev/paddle:0.10.0rc1-cpu /bin/bash

或者，可以以后台进程方式运行容器：

.. code-block:: bash

    docker run -d -p 2202:22 -p 8888:8888 paddledev/paddle:0.10.0rc1-cpu

然后用密码 :code:`root` SSH进入容器：

.. code-block:: bash

    ssh -p 2202 root@localhost

SSH方式的一个优点是我们可以从多个终端进入容器。比如，一个终端运行vi，另一个终端运行Python。另一个好处是我们可以把PaddlePaddle容器运行在远程服务器上，并在笔记本上通过SSH与其连接。


以上方法在GPU镜像里也能用－只是请不要忘记按装CUDA驱动，以及告诉Docker：

.. code-block:: bash

    export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
    export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
    docker run ${CUDA_SO} ${DEVICES} -it paddledev/paddle:0.10.0rc1-gpu


运行PaddlePaddle书籍
---------------------

Jupyter Notebook是一个开源的web程序，大家可以通过它制作和分享带有代码、公式、图表、文字的交互式文档。用户可以通过网页浏览文档。

PaddlePaddle书籍是为用户和开发者制作的一个交互式的Jupyter Nodebook。
如果您想要更深入了解deep learning，PaddlePaddle书籍一定是您最好的选择。

当您进入容器内之后，只用运行以下命令：

.. code-block:: bash
        
    jupyter notebook

然后在浏览器中输入以下网址：
    
.. code-block:: text

    http://localhost:8888/

就这么简单，享受您的旅程！


非AVX镜像
---------

纯CPU镜像以及GPU镜像都会用到AVX指令集，但是2008年之前生产的旧电脑不支持AVX。以下指令能检查Linux电脑是否支持AVX：

.. code-block:: bash

   if cat /proc/cpuinfo | grep -i avx; then echo Yes; else echo No; fi

如果输出是No，我们就需要手动编译一个非AVX版本的镜像：

.. code-block:: bash

   cd ~
   git clone https://github.com/PaddlePaddle/Paddle.git
   cd Paddle
   docker build --build-arg WITH_AVX=OFF -t paddle:cpu-noavx -f paddle/scripts/docker/Dockerfile .
   docker build --build-arg WITH_AVX=OFF -t paddle:gpu-noavx -f paddle/scripts/docker/Dockerfile.gpu .


通过Docker容器开发PaddlePaddle
------------------------------

开发人员可以在Docker中开发PaddlePaddle。这样开发人员可以以一致的方式在不同的平台上工作 - Linux，Mac OS X和Windows。

1. 将开发环境构建为Docker镜像
   
   .. code-block:: bash

      git clone --recursive https://github.com/PaddlePaddle/Paddle
      cd Paddle
      docker build -t paddle:dev -f paddle/scripts/docker/Dockerfile .


   请注意，默认情况下，:code:`docker build` 不会将源码导入到镜像中并编译它。如果我们想这样做，需要设置一个参数：

   .. code-block:: bash

      docker build -t paddle:dev -f paddle/scripts/docker/Dockerfile --build-arg BUILD_AND_INSTALL=ON .


2. 运行开发环境

   当我们编译好了 :code:`paddle:dev`， 我们可以在docker容器里做开发，源代码可以通过挂载本地文件来被载入Docker的开发环境里面：
   
   .. code-block:: bash

      docker run -d -p 2202:22 -v $PWD:/paddle paddle:dev

   以上代码会启动一个带有PaddlePaddle开发环境的docker容器，源代码会被挂载到 :code:`/paddle` 。

   请注意， :code:`paddle:dev` 的默认入口是 :code:`sshd` 。以上的 :code:`docker run` 命令其实会启动一个在2202端口监听的SSHD服务器。这样，我们就能SSH进入我们的开发容器了：
   
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

Paddle的Docker镜像带有一个通过 `woboq code browser
<https://github.com/woboq/woboq_codebrowser>`_ 生成的HTML版本的C++源代码，便于用户浏览C++源码。

只要在Docker里启动PaddlePaddle的时候给它一个名字，就可以再运行另一个Nginx Docker镜像来服务HTML代码：

.. code-block:: bash

   docker run -d --name paddle-cpu-doc paddle:0.10.0rc1-cpu
   docker run -d --volumes-from paddle-cpu-doc -p 8088:80 nginx

接着我们就能够打开浏览器在 http://localhost:8088/paddle/ 浏览代码。
