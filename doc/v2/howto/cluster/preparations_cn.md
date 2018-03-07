## 环境准备

1. 准备您的计算集群。计算集群通常由一组（几台到几千台规模）的Linux服务器组成。服务器之间可以通过局域网（LAN）联通，每台服务器具有集群中唯一的IP地址（或者可被DNS解析的主机名）。集群中的每台计算机通常被成为一个“节点”。
1. 我们需要在集群的所有节点上安装 PaddlePaddle。 如果要启用GPU，还需要在节点上安装对应的GPU驱动以及CUDA。PaddlePaddle的安装可以参考[build_and_install](http://www.paddlepaddle.org/docs/develop/documentation/zh/getstarted/build_and_install/index_cn.html)的多种安装方式。我们推荐使用[Docker](http://www.paddlepaddle.org/docs/develop/documentation/zh/getstarted/build_and_install/docker_install_cn.html)安装方式来快速安装PaddlePaddle。

安装完成之后，执行下面的命令可以查看已经安装的版本（docker安装方式可以进入docker容器执行：`docker run -it paddlepaddle/paddle:[tag] /bin/bash`）：
```bash
$ paddle version
PaddlePaddle 0.10.0, compiled with
    with_avx: ON
    with_gpu: OFF
    with_double: OFF
    with_python: ON
    with_rdma: OFF
    with_timer: OFF
```
