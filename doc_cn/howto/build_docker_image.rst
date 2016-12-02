构建PaddlePaddle的Docker Image
==============================
PaddlePaddle的Docker Image构建源码放置在 ``${源码根目录}/paddle/scripts/docker/`` 目录下。该目录有三类文件：

- Dockerfile：Docker Image的描述文件，包括构建步骤、各种参数和维护人员等。
  
  - 一共维护了12个Dockerfile，Dockerfile.m4是它们的模板。
  - PaddlePaddle中所有的Image都基于ubuntu 14.04。

- build.sh：Docker Image的构建脚本，使用方式见下一小节。
- generate.sh：通过Dockerfile.m4模板生成不同的Dockerfile。

使用脚本构建Docker Image
------------------------

进入源码目录，执行 ``docker build`` 命令，即可在本地编译出PaddlePaddle的镜像。简单的使用样例为

..  code-block:: bash

    cd ${源码根目录}/paddle/scripts/docker/
    docker build --build-arg LOWEST_DL_SPEED=50K \
                 --build-arg WITH_GPU=ON \
                 --tag  paddle_gpu:latest

其中，``--build-arg`` 传入的配置参数包括:

- LOWEST\_DL\_SPEED\: 在多线程下载过程中，设置下载线程的最低速度。

  - 默认单位是Bytes，但可以传入10K、10M、或10G等这样的单位。
  - 如果小于这个速度，那么这个线程将会关闭。当所有的线程都关闭了，那么下载进程将会重启。
-  WITH\_GPU\: ON or OFF，是否开启GPU功能。注意，
  - **编译** PaddlePaddle的GPU版本 **不一定** 要在具有GPU的机器上进行。
  - **运行** PaddlePaddle的GPU版本 **一定** 要在具有GPU的机器上运行。

注意：所有Image的构建在Docker 1.12版本测试通过, 低于1.12的版本并没有测试。原因是旧版本可能缺乏 ``--build-arg`` 参数，从而不能在运行编译命令的时候接受参数。
