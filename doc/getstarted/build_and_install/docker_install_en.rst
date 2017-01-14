PaddlePaddle in Docker Containers
=================================

Docker container is currently the only officially-supported way to
running PaddlePaddle.  This is reasonable as Docker now runs on all
major operating systems including Linux, Mac OS X, and Windows.
Please be aware that you will need to change `Dockers settings
<https://github.com/PaddlePaddle/Paddle/issues/627>`_ to make full use
of your hardware resource on Mac OS X and Windows.


Development Using Docker
------------------------

Developers can work on PaddlePaddle using Docker.  This allows
developers to work on different platforms -- Linux, Mac OS X, and
Windows -- in a consistent way.

The general development workflow with Docker and CMake is as follows:

1. Get the source code of Paddle:

   .. code-block:: bash

      git clone https://github.com/PaddlePaddle/Paddle.git


2. Build a development Docker image :code:`paddle:dev` from the source
   code.  This image contains all the development tools and
   dependencies of PaddlePaddle.

   .. code-block:: bash

      cd paddle
      docker build -t paddle:dev -f paddle/scripts/docker/Dockerfile .

   Sometimes docker build might suffer from a slow network connection to the official Ubuntu apt-source servers. In such case, we can specify an apt-source mirror server that is geologically nearer to us. In the following example, we specified an apt-source server that responds fast in China.You can specify the UBUNTU MIRROR with :code:`--build-arg UBUNTU_MIRROR` like the example below.

   .. code-block:: bash

      docker build \
       --build-arg UBUNTU_MIRROR="http://mirrors.163.com" \
       -t paddle:dev \
       -f paddle/scripts/docker/Dockerfile .


3. Run the image as a container and mounting local source code
   directory into the container.  This allows us to change the code on
   the host and build it within the container.

   .. code-block:: bash

      docker run       \
       -d              \
       --name paddle   \
       -p 2022:22      \
       -v $PWD:/paddle \
       paddle:dev

   where :code:`-d` makes the container running in background,
   :code:`--name paddle` allows us to run a nginx container to serve
   documents in this container, :code:`-p 2022:22` allows us to SSH
   into this container, :code:`-v $PWD:/paddle` shares the source code
   on the host with the container.

4. SSH into the container:

   .. code-block:: bash

      ssh root@localhost -p 2022

5. We can edit the source code in the container or on this host.  Then
   we can build using cmake

   .. code-block:: bash

      cd /paddle # where paddle source code has been mounted into the container
      mkdir -p build
      cd build
      cmake -DWITH_TESTING=ON ..
      make -j `nproc`
      CTEST_OUTPUT_ON_FAILURE=1 ctest


CPU-only and GPU Images
-----------------------

For each version of PaddlePaddle, we release 2 Docker images, a
CPU-only one and a CUDA GPU one.  We do so by configuring
`dockerhub.com <https://hub.docker.com/r/paddledev/paddle/>`_
automatically runs the following commands:

.. code-block:: bash

   docker build -t paddle:cpu -f paddle/scripts/docker/Dockerfile .
   docker build -t paddle:gpu -f paddle/scripts/docker/Dockerfile.gpu .


To run the CPU-only image as an interactive container:

.. code-block:: bash

    docker run -it --rm paddledev/paddle:cpu-latest /bin/bash

or, we can run it as a daemon container

.. code-block:: bash

    docker run -d -p 2202:22 paddledev/paddle:cpu-latest

and SSH to this container using password :code:`root`:

.. code-block:: bash

    ssh -p 2202 root@localhost

An advantage of using SSH is that we can connect to PaddlePaddle from
more than one terminals.  For example, one terminal running vi and
another one running Python interpreter.  Another advantage is that we
can run the PaddlePaddle container on a remote server and SSH to it
from a laptop.


Above methods work with the GPU image too -- just please don't forget
to install CUDA driver and let Docker knows about it:

.. code-block:: bash

    export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
    export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
    docker run ${CUDA_SO} ${DEVICES} -it paddledev/paddle:gpu-latest


Non-AVX Images
--------------

Please be aware that the CPU-only and the GPU images both use the AVX
instruction set, but old computers produced before 2008 do not support
AVX.  The following command checks if your Linux computer supports
AVX:

.. code-block:: bash

   if cat /proc/cpuinfo | grep -i avx; then echo Yes; else echo No; fi


If it doesn't, we will need to build non-AVX images manually from
source code:

.. code-block:: bash

   cd ~
   git clone https://github.com/PaddlePaddle/Paddle.git
   cd Paddle
   docker build --build-arg WITH_AVX=OFF -t paddle:cpu-noavx -f paddle/scripts/docker/Dockerfile .
   docker build --build-arg WITH_AVX=OFF -t paddle:gpu-noavx -f paddle/scripts/docker/Dockerfile.gpu .


Documentation
-------------

Paddle Docker images include an HTML version of C++ source code
generated using `woboq code browser
<https://github.com/woboq/woboq_codebrowser>`_.  This makes it easy
for users to browse and understand the C++ source code.

As long as we give the Paddle Docker container a name, we can run an
additional Nginx Docker container to serve the volume from the Paddle
container:

.. code-block:: bash

   docker run -d --name paddle-cpu-doc paddle:cpu
   docker run -d --volumes-from paddle-cpu-doc -p 8088:80 nginx


Then we can direct our Web browser to the HTML version of source code
at http://localhost:8088/paddle/
