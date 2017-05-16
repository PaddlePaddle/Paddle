构建PaddlePaddle Docker Image
===========================

PaddlePaddle的Docker Image构建源码放置在 :code:`${源码根目录}/paddle/scripts/docker/`目录下。
该Image基于ubuntu 14.04。该目录下有两个文件，Dockerfile和build.sh。其中:

*  Dockerfile是docker image的主要描述文件。描述了Docker image的构建步骤、各种参数和维护
   人员等等。
*  build.sh是docker image的主要构建步骤。

该image的构建在docker 1.12版本测试通过, 低于docker 1.12版本的情况下并没有测试。主要由于旧版本
的docker可能缺乏 :code:`--build-arg` 参数，从而不能在运行编译命令的时候接受参数。

同时，该构建脚本充分考虑了网络不稳定的情况，对于cuda的Toolkit有断点续传和传输速度过小重启下载的
简单优化。

使用脚本构建PaddlePaddle Docker Image
-------------------------------------------

该脚本的使用方法是，进入该源码目录，执行 :code:`docker build .` 命令。可以使用
 :code:`--build-arg` 传入的配置参数包括:

*  LOWEST\_DL\_SPEED\: 多线程下载过程中，最低线程的下载速度(默认单位是Bytes，可以传入10K, 
   10M，或者10G这样的单位)。如果小于这个下载速度，那么这个下载线程将会关闭。所有的下载线程关闭时，
   下载进程会重启。
*  WITH\_GPU\: ON or OFF。是否开启GPU功能。注意，编译PaddlePaddle的GPU版本并不需要一定在具有GPU
   的机器上进行。但是，运行PaddlePaddle的GPU版本一定要在具有CUDA的机器上运行。

简单的使用样例为\:

..  code-block:: bash

    cd ${源码根目录}/paddle/scripts/docker/
    docker build --build-arg LOWEST_DL_SPEED=50K\
                 --build-arg WITH_GPU=ON \
                 --tag  paddle_gpu:latest .

即可在本地编译出PaddlePaddle的镜像。
