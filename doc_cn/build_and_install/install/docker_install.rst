安装PaddlePaddle的Docker镜像
============================

PaddlePaddle提供了Docker的使用镜像。PaddlePaddle推荐使用Docker进行PaddlePaddle的部署和
运行。Docker是一个基于容器的轻量级虚拟环境。具有和宿主机相近的运行效率，并提供
了非常方便的二进制分发手段。

下述内容将分为如下几个类别描述。

* PaddlePaddle提供的Docker镜像版本
* 下载和运行Docker镜像
* 注意事项

PaddlePaddle提供的Docker镜像版本
--------------------------------

我们提供了12个 `Docker image <https://hub.docker.com/r/paddledev/paddle/tags/>`_ ，他们的image name都是 :code:`paddle-dev/paddle` ，tag分别为

+-----------------+------------------+------------------------+-----------------------+
|                 |   normal         |           devel        |          demo         |
+=================+==================+========================+=======================+
|       CPU       | cpu-latest       | cpu-devel-latest       | cpu-demo-latest       |
+-----------------+------------------+------------------------+-----------------------+
|       GPU       | gpu-latest       | gpu-devel-latest       | gpu-demo-latest       |
+-----------------+------------------+------------------------+-----------------------+
| CPU WITHOUT AVX | cpu-noavx-latest | cpu-noavx-devel-latest | cpu-noavx-demo-latest |
+-----------------+------------------+------------------------+-----------------------+
| GPU WITHOUT AVX | gpu-noavx-latest | gpu-noavx-devel-latest | gpu-noavx-demo-latest |
+-----------------+------------------+------------------------+-----------------------+

其中，横向包括三个版本，normal，devel和demo。

* Normal: 正常的Docker image，只包括paddle的二进制
* Devel: 包括Paddle的二进制、编译环境和源代码
* Demo: 包括Paddle运行demo所需要的依赖

纵向包括四个版本，他们是。

* CPU: CPU版本。需要支持AVX指令集的CPU
* GPU: GPU版本。需要支持AVX指令集的CPU
* CPU WITHOUT AVX: CPU版本，不支持AVX指令集的CPU也可以运行
* GPU WITHOUT AVX: GPU版本，不需要AVX指令集的CPU也可以运行。

用户可以选择对应版本的docker image。使用如下脚本可以确定本机的CPU知否支持 :code:`AVX` 指令集\:

..  code-block:: bash

    if cat /proc/cpuinfo | grep -q avx ; then echo "Support AVX"; else echo "Not support AVX"; fi

如果输出 :code:`Support AVX`，则可以选择上表中的AVX版本PaddlePaddle。否则需要选择非AVX的PaddlePaddle。选择普通CPU版本的devel版本的image，则可以使用 :code:`paddle-dev/paddle:cpu-devel-latest` 来引用这个image。

PaddlePaddle提供的镜像并不包含任何命令运行，想要运行PaddlePaddle，您需要进入镜像运行PaddlePaddle
程序或者自定义一个含有启动脚本的image。具体请参考注意事项中的 :code:`使用ssh访问PaddlePaddle镜像`

下载和运行Docker镜像
--------------------

为了运行PaddlePaddle的docker镜像，您需要在机器中安装好Docker。安装Docker需要您的机器
至少具有3.10以上的linux kernel。安装方法请参考
`Docker的官方文档 <https://docs.docker.com/engine/installation/>`_ 。如果您使用
mac osx或者是windows机器，请参考 
`mac osx的安装文档 <https://docs.docker.com/engine/installation/mac/>`_ 和
`windows 的安装文档 <https://docs.docker.com/engine/installation/windows/>`_ 。

您可以使用 :code:`docker pull` 命令预先下载镜像，也可以直接执行 
:code:`docker run` 命令运行镜像。执行方法如下:

..  code-block:: bash
    
    $ docker run -it paddledev/paddlepaddle:cpu-latest

即可启动和进入PaddlePaddle的container。如果运行GPU版本的PaddlePaddle，则需要先将
cuda相关的Driver和设备映射进container中，脚本类似于

..  code-block:: bash

    $ export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
    $ export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
    $ docker run ${CUDA_SO} ${DEVICES} -it paddledev/paddlepaddle:latest-gpu

进入Docker container后，运行 :code:`paddle version` 即可打印出PaddlePaddle的版本和构建
信息。安装完成的PaddlePaddle主体包括三个部分， :code:`paddle` 脚本， python的
:code:`paddle` 包和 :code:`py_paddle` 包。其中\:

* :code:`paddle` 脚本和 :code:`paddle` 的python包是PaddlePaddle的训练主要程序。使用 
  :code:`paddle` 脚本可以启动PaddlePaddle的训练进程和pserver。而 :code:`paddle` 脚本
  中的二进制使用了 :code:`paddle` 的python包来做配置文件解析等工作。
* python包 :code:`py_paddle` 是一个swig封装的PaddlePaddle包，用来做预测和简单的定制化
  训练。

注意事项
--------

性能问题
++++++++

由于Docker是基于容器的轻量化虚拟方案，所以在CPU的运算性能上并不会有严重的影响。
而GPU的驱动和设备全部映射到了容器内，所以GPU在运算性能上也不会有严重的影响。

但是如果使用了高性能的网卡，例如RDMA网卡(RoCE 40GbE 或者 IB 56GbE)，或者高性能的
以太网卡 (10GbE)。推荐使用将本地网卡，即 "--net=host" 来进行训练。而不使用docker
的网桥来进行网络通信。

远程访问问题和二次开发
++++++++++++++++++++++

由于PaddlePaddle的Docker镜像并不包含任何预定义的运行命令。所以如果想要在后台启用ssh
远程访问，则需要进行一定的二次开发，将ssh装入系统内并开启远程访问。二次开发可以
使用Dockerfile构建一个全新的docker image。需要参考 
`Dockerfile的文档 <https://docs.docker.com/engine/reference/builder/>`_ 和
`Dockerfile的最佳实践 <https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/>`_ 
两个文档。

简单的含有ssh的Dockerfile如下：

..  literalinclude:: paddle_ssh.Dockerfile

使用该Dockerfile构建出镜像，然后运行这个container即可。相关命令为\:

..  code-block:: bash

    # cd到含有Dockerfile的路径中
    $ docker build . -t paddle_ssh
    # 运行这个container，将宿主机的8022端口映射到container的22端口上
    $ docker run -d -p 8022:22  --name paddle_ssh_machine paddle_ssh

执行如下命令即可以关闭这个container，并且删除container中的数据\:

..  code-block:: bash
    
    # 关闭container
    $ docker stop paddle_ssh_machine
    # 删除container
    $ docker rm paddle_ssh_machine

如果想要在外部机器访问这个container，即可以使用ssh访问宿主机的8022端口。用户名为
root，密码也是root。命令为\:

..  code-block:: bash

    $ ssh -p 8022 root@YOUR_HOST_MACHINE

至此，您就可以远程的使用PaddlePaddle啦。
