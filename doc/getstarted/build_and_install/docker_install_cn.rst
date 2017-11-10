PaddlePaddle的Docker容器使用方式
================================

PaddlePaddle目前唯一官方支持的运行的方式是Docker容器。因为Docker能在所有主要操作系统（包括Linux，Mac OS X和Windows）上运行。 请注意，您需要更改 `Dockers设置 <https://github.com/PaddlePaddle/Paddle/issues/627>`_ 才能充分利用Mac OS X和Windows上的硬件资源。

Docker使用入门
------------------------------

几个基础的概念帮助理解和使用Docker：

- *镜像*：一个Docker镜像是一个打包好的软件。它包含了这个软件本身和它所依赖的运行环境。PaddlePaddle的Docker镜像就包含了PaddlePaddle的Python库以及其依赖的多个Python库。这样我们可以直接在Docker中运行需要的程序而不需要安装后在执行。可以执行：

  .. code-block:: bash

     docker images

  来列出当前系统中的所有镜像，同样可以执行：

  .. code-block:: bash
		  
     docker pull paddlepaddle/paddle:0.10.0

  来下载Docker镜像，paddlepaddle/paddle是从官方镜像源Dockerhub.com下载的，推荐国内用户使用docker.paddlepaddle.org/paddle下载。

- *容器*： 如果说一个Docker镜像就是一个程序，那容器就是这个程序运行时产生的“进程”。
  实际上，一个容器就是一个操作系统的进程，但是是运行在独立的进程空间，文件系统以及网络之上。
  可以执行：

  .. code-block:: bash

     docker run paddlepaddle/paddle:0.10.0

  来使用一个镜像启动一个容器。

- 默认情况下，Docker容器会运行在独立的文件系统空间之上，我们无法在Docker容器中
  访问到主机上的文件。可以通过*挂载Volume*的方式，将主机上的文件或目录挂载到
  Docker容器中。下面的命令把当前目录挂载到了容器中的 /data 目录下，容器使用
  debian镜像，并且启动后执行 :code:`ls /data`。

  .. code-block:: bash

     docker run --rm -v $(pwd):/data debian ls /data

PaddlePaddle发布的Docker镜像使用说明
------------------------------

我们把PaddlePaddle的编译环境打包成一个镜像，称为开发镜像，里面涵盖了
PaddlePaddle需要的所有编译工具。把编译出来的PaddlePaddle也打包成一个镜
像，称为生产镜像，里面涵盖了PaddlePaddle运行所需的所有环境。每次
PaddlePaddle发布新版本的时候都会发布对应版本的生产镜像以及开发镜像。运
行镜像包括纯CPU版本和GPU版本以及其对应的非AVX版本。我们会在
`dockerhub.com <https://hub.docker.com/r/paddlepaddle/paddle/tags/>`_ 
和国内镜像`docker.paddlepaddle.org` 提供最新
的Docker镜像，可以在"tags"标签下找到最新的Paddle镜像版本。

**注意：为了方便在国内的开发者下载Docker镜像，我们提供了国内的镜像服务器供大家使用。如果您在国内，请把文档里命令中的paddlepaddle/paddle替换成docker.paddlepaddle.org/paddle。**

1. 开发镜像：:code:`paddlepaddle/paddle:0.10.0-dev`

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

      docker run -it --rm -v $(pwd):/paddle  paddlepaddle/paddle:0.10.0-dev /bin/bash

   或者，可以以后台进程方式运行容器：

   .. code-block:: bash

      docker run -d -p 2202:22 -p 8888:8888 -v $(pwd):/paddle paddlepaddle/paddle:0.10.0-dev /usr/sbin/sshd -D

   然后用密码 :code:`root` SSH进入容器：

   .. code-block:: bash

      ssh -p 2202 root@localhost

   SSH方式的一个优点是我们可以从多个终端进入容器。比如，一个终端运行vi，另一个终端运行Python。另一个好处是我们可以把PaddlePaddle容器运行在远程服务器上，并在笔记本上通过SSH与其连接。

2. 生产镜像：根据CPU、GPU和非AVX区分了如下4个镜像：

   - GPU/AVX：:code:`paddlepaddle/paddle:<version>-gpu`
   - GPU/no-AVX：:code:`paddlepaddle/paddle:<version>-gpu-noavx`
   - CPU/AVX：:code:`paddlepaddle/paddle:<version>`
   - CPU/no-AVX：:code:`paddlepaddle/paddle:<version>-noavx`

   纯CPU镜像以及GPU镜像都会用到AVX指令集，但是2008年之前生产的旧电脑不支持AVX。以下指令能检查Linux电脑是否支持AVX：

   .. code-block:: bash

      if cat /proc/cpuinfo | grep -i avx; then echo Yes; else echo No; fi

   如果输出是No，就需要选择使用no-AVX的镜像

   **注：在0.10.0之后的版本，PaddlePaddle都可以自动判断硬件是否支持AVX，所以无需判断AVX即可使用**

   以上方法在GPU镜像里也能用，只是请不要忘记提前在物理机上安装GPU最新驱动。
   为了保证GPU驱动能够在镜像里面正常运行，我们推荐使用[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)来运行镜像。

   .. code-block:: bash

      nvidia-docker run -it --rm paddledev/paddle:0.10.0-gpu /bin/bash

   注意: 如果使用nvidia-docker存在问题，你也许可以尝试更老的方法，具体如下，但是我们并不推荐这种方法。：

   .. code-block:: bash

      export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
      export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
      docker run ${CUDA_SO} ${DEVICES} -it paddledev/paddle:0.10.0-gpu

3. 运行以及发布您的AI程序

   假设您已经完成了一个AI训练的python程序 :code:`a.py`，这个程序是您在开发机上使用开发镜像完成开发。此时您可以运行这个命令在开发机上进行测试运行：

   .. code-block:: bash

      docker run -it -v $PWD:/work paddle /work/a.py

   如果要使用GPU，请运行：

   .. code-block:: bash

      nvidia-docker run -it -v $PWD:/work paddle /work/a.py


   这里`a.py`包含的所有依赖假设都可以在Paddle的运行容器中。如果需要包含更多的依赖、或者需要发布您的应用的镜像，可以编写`Dockerfile`使用`FROM paddledev/paddle:0.10.0`
   创建和发布自己的AI程序镜像。

运行PaddlePaddle Book
---------------------

Jupyter Notebook是一个开源的web程序，大家可以通过它制作和分享带有代码、公式、图表、文字的交互式文档。用户可以通过网页浏览文档。

PaddlePaddle Book是为用户和开发者制作的一个交互式的Jupyter Notebook。
如果您想要更深入了解deep learning，PaddlePaddle Book一定是您最好的选择。

我们提供可以直接运行PaddlePaddle Book的Docker镜像，直接运行：

.. code-block:: bash

    docker run -p 8888:8888 paddlepaddle/book

然后在浏览器中输入以下网址：

.. code-block:: text

    http://localhost:8888/

就这么简单，享受您的旅程！

通过Docker容器开发PaddlePaddle
------------------------------

开发人员可以在Docker开发镜像中开发PaddlePaddle。这样开发人员可以以一致的方式在不同的平台上工作 - Linux，Mac OS X和Windows。

1. 制作PaddlePaddle开发镜像

   PaddlePaddle每次发布新版本都会发布对应的开发镜像供开发者直接使用。这里介绍如生成造这个开发镜像。
   生成Docker镜像的方式有两个，一个是直接把一个容器转换成镜像，另一个是创建Dockerfile并运行docker build指令按照Dockerfile生成镜像。第一个方法的好处是简单快捷，适合自己实验，可以快速迭代。第二个方法的好处是Dockerfile可以把整个生成流程描述很清楚，其他人很容易看懂镜像生成过程，持续集成系统也可以简单地复现这个过程。我们采用第二个方法。Dockerfile位于PaddlePaddle repo的根目录。生成生产镜像只需要运行：

   .. code-block:: bash
      
      git clone https://github.com/PaddlePaddle/Paddle.git
      cd Paddle
      docker build -t paddle:dev .

   docker build这个命令的-t指定了生成的镜像的名字，这里我们用paddle:dev。到此，PaddlePaddle开发镜像就被构建完毕了。

2. 制作PaddlePaddle生产镜像

   生产镜像的生成分为两步，第一步是运行：

   .. code-block:: bash
      
      docker run -v $(pwd):/paddle -e "WITH_GPU=OFF" -e "WITH_AVX=OFF" -e "WITH_TEST=ON" paddle:dev

   以上命令会编译PaddlePaddle，生成运行程序，以及生成创建生产镜像的Dockerfile。所有生成的的文件都在build目录下。“WITH_GPU”控制生成的生产镜像是否支持GPU，“WITH_AVX”控制生成的生产镜像是否支持AVX，”WITH_TEST“控制是否生成单元测试。

   第二步是运行：

   .. code-block:: bash
      
      docker build -t paddle:prod -f build/Dockerfile ./build

   以上命令会按照生成的Dockerfile把生成的程序拷贝到生产镜像中并做相应的配置，最终生成名为paddle:prod的生产镜像。

3. 运行单元测试

   运行以下指令：

   .. code-block:: bash
      
      docker run -it -v $(pwd):/paddle paddle:dev bash -c "cd /paddle/build && ctest"

文档
----

Paddle的Docker开发镜像带有一个通过 `woboq code browser
<https://github.com/woboq/woboq_codebrowser>`_ 生成的HTML版本的C++源代码，便于用户浏览C++源码。

只要在Docker里启动PaddlePaddle的时候给它一个名字，就可以再运行另一个Nginx Docker镜像来服务HTML代码：

.. code-block:: bash

   docker run -d --name paddle-cpu-doc paddle:0.10.0-dev
   docker run -d --volumes-from paddle-cpu-doc -p 8088:80 nginx

接着我们就能够打开浏览器在 http://localhost:8088/paddle/ 浏览代码。
