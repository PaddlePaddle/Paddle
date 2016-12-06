PaddlePaddle in Docker Containers
=================================

Docker container is currently the only officially-supported way to
running PaddlePaddle.  This is reasonable as Docker now runs on all
major operating systems including Linux, Mac OS X, and Windows.
Please be aware that you will need to change `Dockers settings
<https://github.com/PaddlePaddle/Paddle/issues/627>`_ to make full use
of your hardware resource on Mac OS X and Windows.


CPU-only and GPU Images
-----------------------

For each version of PaddlePaddle, we release 2 Docker images, a
CPU-only one and a CUDA GPU one.  We do so by configuring
`dockerhub.com <https://hub.docker.com/r/paddledev/paddle/>`_
automatically runs the following commands:

.. code-block:: base

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
   git clone github.com/PaddlePaddle/Paddle
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
additional nginx Docker container to serve the volume from the Paddle
container:

.. code-block:: bash

   docker run -d --name paddle-cpu-doc paddle:cpu
   docker run -d --volumes-from paddle-cpu-doc -p 8088:80 nginx


Then we can direct our Web browser to the HTML version of source code
at http://localhost:8088/paddle/
