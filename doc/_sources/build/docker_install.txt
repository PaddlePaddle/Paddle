Docker installation guide
==========================

PaddlePaddle provide the `Docker <https://www.docker.com/>`_ image. `Docker`_ is a lightweight container utilities. The performance of PaddlePaddle in `Docker`_ container is basically as same as run it in a normal linux. The `Docker`_ is a very convenient way to deliver the binary release for linux programs.

..  note::

    The `Docker`_ image is the recommended way to run PaddlePaddle 

PaddlePaddle Docker images
--------------------------

There are 12 `images <https://hub.docker.com/r/paddledev/paddle/tags/>`_ for PaddlePaddle, and the name is :code:`paddle-dev/paddle`,  tags are\: 


+-----------------+------------------+------------------------+-----------------------+
|                 |   normal         |           devel        |          demo         |
+=================+==================+========================+=======================+
|       CPU       | cpu-latest       | cpu-devel-latest       | cpu-demo-latest       |
+-----------------+------------------+------------------------+-----------------------+
|       GPU       | gpu-latest       | gpu-devel-latest       | gpu-demo-latest       |
+-----------------+------------------+------------------------+-----------------------+
| CPU WITHOUT AVX | cpu-noavx-latest | cpu-devel-noavx-latest | cpu-demo-noavx-latest |
+-----------------+------------------+------------------------+-----------------------+
| GPU WITHOUT AVX | gpu-noavx-latest | gpu-devel-noavx-latest | gpu-demo-noavx-latest |
+-----------------+------------------+------------------------+-----------------------+

And the three columns are:

* normal\: The docker image only contains binary of PaddlePaddle.
* devel\: The docker image contains PaddlePaddle binary, source code and essential build environment.
* demo\: The docker image contains the dependencies to run PaddlePaddle demo.

And the four rows are:

* CPU\: CPU Version. Support CPU which has :code:`AVX` instructions.
* GPU\: GPU Version. Support GPU, and cpu has :code:`AVX` instructions.
* CPU WITHOUT AVX\: CPU Version, which support most CPU even doesn't have :code:`AVX` instructions.
* GPU WITHOUT AVX\: GPU Version, which support most CPU even doesn't have :code:`AVX` instructions.

User can choose any version depends on machine. The following script can help you to detect your CPU support :code:`AVX` or not.

..  code-block:: bash
    
    if cat /proc/cpuinfo | grep -q avx ; then echo "Support AVX"; else echo "Not support AVX"; fi

If the output is :code:`Support AVX`, then you can choose the AVX version of PaddlePaddle, otherwise, you need select :code:`noavx` version of PaddlePaddle. For example, the CPU develop version of PaddlePaddle is :code:`paddle-dev/paddle:cpu-devel-latest`.

The PaddlePaddle images don't contain any entry command. You need to write your entry command to use this image. See :code:`Remote Access` part or just use following command to run a :code:`bash`

..  code-block:: bash

    docker run -it paddledev/paddle:cpu-latest /bin/bash


Download and Run Docker images
------------------------------

You have to install Docker in your machine which has linux kernel version 3.10+ first. You can refer to the official guide https://docs.docker.com/engine/installation/ for further information.

You can use :code:`docker pull ` to download images first, or just launch a container with :code:`docker run` \:

..  code-block:: bash

    docker run -it paddledev/paddle:cpu-latest


If you want to launch container with GPU support, you need to set some environment variables at the same time:

..  code-block:: bash

    export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
    export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
    docker run ${CUDA_SO} ${DEVICES} -it paddledev/paddle:gpu-latest


Some notes for docker
---------------------

Performance
+++++++++++

Since Docker is based on the lightweight virtual containers, the CPU computing performance maintains well. And GPU driver and equipments are all mapped to the container, so the GPU computing performance would not be seriously affected.

If you use high performance nic, such as RDMA(RoCE 40GbE or IB 56GbE), Ethernet(10GbE), it is recommended to use config "-net = host".




Remote access
+++++++++++++


If you want to enable ssh access background, you need to build an image by yourself. Please refer to official guide https://docs.docker.com/engine/reference/builder/ for further information.

Following is a simple Dockerfile with ssh:

..  literalinclude:: ../../doc_cn/build_and_install/install/paddle_ssh.Dockerfile

Then you can build an image with Dockerfile and launch a container:

..  code-block:: bash

    # cd into Dockerfile directory
    docker build . -t paddle_ssh
    # run container, and map host machine port 8022 to container port 22
    docker run -d -p 8022:22 --name paddle_ssh_machine paddle_ssh

Now, you can ssh on port 8022 to access the container, username is root, password is also root:

..  code-block:: bash

    ssh -p 8022 root@YOUR_HOST_MACHINE

You can stop and delete the container as following:

..  code-block:: bash

    # stop
    docker stop paddle_ssh_machine
    # delete
    docker rm paddle_ssh_machine
