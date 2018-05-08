## Preparations

1. Prepare your computer cluster. It's normally a bunch of Linux servers connected by LAN. Each server will be assigned a unique IP address. The computers in the cluster can be called "nodes".
2. Install PaddlePaddle on every node. If you are going to take advantage of GPU cards, you'll also need to install proper driver and CUDA libraries. To install PaddlePaddle please read [this build and install](http://www.paddlepaddle.org/docs/develop/documentation/en/getstarted/build_and_install/index_en.html) document. We strongly recommend using [Docker installation](http://www.paddlepaddle.org/docs/develop/documentation/en/getstarted/build_and_install/docker_install_en.html).

After installation, you can check the version by typing the below command (run a docker container  if using docker: `docker run -it paddlepaddle/paddle:[tag] /bin/bash`):

```bash
$ paddle version
PaddlePaddle 0.10.0rc, compiled with
    with_avx: ON
    with_gpu: OFF
    with_double: OFF
    with_python: ON
    with_rdma: OFF
    with_timer: OFF
```
