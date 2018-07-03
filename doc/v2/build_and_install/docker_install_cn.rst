使用Docker安装运行
================================

使用Docker安装和运行PaddlePaddle可以无需考虑依赖环境即可运行。并且也可以在Windows的docker中运行。
您可以在 `Docker官网 <https://docs.docker.com/get-started/>`_ 获得基本的Docker安装和使用方法。

如果您在使用Windows，可以参考
`这篇 <https://docs.docker.com/toolbox/toolbox_install_windows/>`_
教程，完成在Windows上安装和使用Docker。

在了解Docker的基本使用方法之后，即可开始下面的步骤：

.. _docker_pull:

获取PaddlePaddle的Docker镜像
------------------------------

执行下面的命令获取最新的PaddlePaddle Docker镜像，版本为cpu_avx_mkl：

  .. code-block:: bash

     docker pull paddlepaddle/paddle

对于国内用户，我们提供了加速访问的镜像源：

  .. code-block:: bash

     docker pull docker.paddlepaddlehub.com/paddle

下载GPU版本（cuda8.0_cudnn5_avx_mkl）的Docker镜像：

  .. code-block:: bash

     docker pull paddlepaddle/paddle:latest-gpu
     docker pull docker.paddlepaddlehub.com/paddle:latest-gpu

选择下载使用不同的BLAS库的Docker镜像：

  .. code-block:: bash

     # 默认是使用MKL的镜像
     docker pull paddlepaddle/paddle
     # 使用OpenBLAS的镜像
     docker pull paddlepaddle/paddle:latest-openblas

下载指定版本的Docker镜像，可以从 `DockerHub网站 <https://hub.docker.com/r/paddlepaddle/paddle/tags/>`_ 获取可选的tag，并执行下面的命令：

  .. code-block:: bash

     docker pull paddlepaddle/paddle:[tag]
     # 比如：
     docker pull docker.paddlepaddlehub.com/paddle:0.11.0-gpu

.. _docker_run:

在Docker中执行PaddlePaddle训练程序
----------------------------------

假设您已经在当前目录（比如在/home/work）编写了一个PaddlePaddle的程序 :code:`train.py` （可以参考
`PaddlePaddleBook <http://www.paddlepaddle.org/docs/develop/book/01.fit_a_line/index.cn.html>`_ 
编写），就可以使用下面的命令开始执行训练：

  .. code-block:: bash

     cd /home/work
     docker run -it -v $PWD:/work paddlepaddle/paddle /work/train.py
 
上述命令中， :code:`-it` 参数说明容器已交互式运行； :code:`-v $PWD:/work`
指定将当前路径（Linux中$PWD变量会展开为当前路径的绝对路径）挂载到容器内部的 :code:`/work`
目录； :code:`paddlepaddle/paddle` 指定需要使用的容器； 最后 :code:`/work/train.py`
为容器内执行的命令，即运行训练程序。

当然，您也可以进入到Docker容器中，以交互式的方式执行或调试您的代码：

  .. code-block:: bash

     docker run -it -v $PWD:/work paddlepaddle/paddle /bin/bash
     cd /work
     python train.py

**注：PaddlePaddle Docker镜像为了减小体积，默认没有安装vim，您可以在容器中执行** :code:`apt-get install -y vim` **安装后，在容器中编辑代码。**

.. _docker_run_book:

使用Docker启动PaddlePaddle Book教程
-----------------------------------

使用Docker可以快速在本地启动一个包含了PaddlePaddle官方Book教程的Jupyter Notebook，可以通过网页浏览。
PaddlePaddle Book是为用户和开发者制作的一个交互式的Jupyter Notebook。
如果您想要更深入了解deep learning，PaddlePaddle Book一定是您最好的选择。
大家可以通过它阅读教程，或者制作和分享带有代码、公式、图表、文字的交互式文档。

我们提供可以直接运行PaddlePaddle Book的Docker镜像，直接运行：

  .. code-block:: bash

     docker run -p 8888:8888 paddlepaddle/book

国内用户可以使用下面的镜像源来加速访问：

  .. code-block:: bash

    docker run -p 8888:8888 docker.paddlepaddlehub.com/book

然后在浏览器中输入以下网址：

  .. code-block:: text

     http://localhost:8888/

就这么简单，享受您的旅程！

.. _docker_run_gpu:

使用Docker执行GPU训练
------------------------------

为了保证GPU驱动能够在镜像里面正常运行，我们推荐使用
`nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ 来运行镜像。
请不要忘记提前在物理机上安装GPU最新驱动。

  .. code-block:: bash

     nvidia-docker run -it -v $PWD:/work paddlepaddle/paddle:latest-gpu /bin/bash

**注: 如果没有安装nvidia-docker，可以尝试以下的方法，将CUDA库和Linux设备挂载到Docker容器内：**

  .. code-block:: bash

     export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
     export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
     docker run ${CUDA_SO} ${DEVICES} -it paddlepaddle/paddle:latest-gpu

**关于AVX：**

AVX是一种CPU指令集，可以加速PaddlePaddle的计算。最新的PaddlePaddle Docker镜像默认
是开启AVX编译的，所以，如果您的电脑不支持AVX，需要单独
`编译 <./build_from_source_cn.html>`_ PaddlePaddle为no-avx版本。

以下指令能检查Linux电脑是否支持AVX：

   .. code-block:: bash

      if cat /proc/cpuinfo | grep -i avx; then echo Yes; else echo No; fi

如果输出是No，就需要选择使用no-AVX的镜像
