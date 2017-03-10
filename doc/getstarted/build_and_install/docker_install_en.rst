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

1. Build the Development Environment as a Docker Image

   .. code-block:: bash

      git clone --recursive https://github.com/PaddlePaddle/Paddle
      cd Paddle
      docker build -t paddle:dev -f paddle/scripts/docker/Dockerfile .


   Note that by default :code:`docker build` wouldn't import source
   tree into the image and build it.  If we want to do that, we need
   to set a build arg:

   .. code-block:: bash

      docker build -t paddle:dev -f paddle/scripts/docker/Dockerfile --build-arg BUILD_AND_INSTALL=ON .


2. Run the Development Environment

   Once we got the image :code:`paddle:dev`, we can use it to develop
   Paddle by mounting the local source code tree into a container that
   runs the image:

   .. code-block:: bash

      docker run -d -p 8888:8888 -v $PWD:/paddle --rm --name mypaddle-dev paddle:dev

   This runs a container of the development environment Docker image
   with the local source tree mounted to :code:`/paddle` of the
   container.

   We can get into this container with:
   .. code-block:: bash
      docker exec -it mypaddle-dev /bin/bash

   Usually, I run above commands on my Mac.  I can also run them on a
   GPU server :code:`xxx.yyy.zzz.www` and run `docker exec` to get into the container.

   .. code-block:: bash
      my-mac$ docker exec -it mypaddle-dev /bin/bash

3. Build and Install Using the Development Environment

   Once I am in the container, I can use
   :code:`paddle/scripts/docker/build.sh` to build, install, and test
   Paddle:

   .. code-block:: bash

      /paddle/paddle/scripts/docker/build.sh

   This builds everything about Paddle in :code:`/paddle/build`.  And
   we can run unit tests there:

   .. code-block:: bash

      cd /paddle/build
      ctest

4. Run PaddlePaddle Book under Docker Container

   The Jupyter Notebook is an open-source web application that allows
   you to create and share documents that contain live code, equations,
   visualizations and explanatory text in a single browser.

   PaddlePaddle Book is an interactive Jupyter Notebook for users and developers.
   We already exposed port 8888 for this book. If you want to
   dig deeper into deep learning, PaddlePaddle Book definitely is your best choice.

   Once you are inside the container, simply issue the command:

   .. code-block:: bash

      jupyter notebook

   Then, you would back and paste the address into the local browser:

   .. code-block:: text

      http://localhost:8888/

   That's all. Enjoy your journey!

CPU-only and GPU Images
-----------------------

For each version of PaddlePaddle, we release 2 Docker images, a
CPU-only one and a CUDA GPU one.  We do so by configuring
`dockerhub.com <https://hub.docker.com/r/paddledev/paddle/>`_
automatically runs the following commands:

.. code-block:: bash

   docker build -t paddle:cpu -f paddle/scripts/docker/Dockerfile --build-arg BUILD_AND_INSTALL=ON .
   docker build -t paddle:gpu -f paddle/scripts/docker/Dockerfile.gpu --build-arg BUILD_AND_INSTALL=ON .


To run the CPU-only image as an interactive container:

.. code-block:: bash

    docker run -it --rm paddledev/paddle:0.10.0rc1-cpu /bin/bash

or, we can run it as a daemon container

.. code-block:: bash

    docker run -d --rm --name paddl-cpu paddledev/paddle:0.10.0rc1-cpu

and get into this container using :code:`docker exec`:

.. code-block:: bash
    docker exec -it paddle-cpu /bin/bash


Above methods work with the GPU image too -- just please don't forget
to install CUDA driver and let Docker knows about it:

.. code-block:: bash

    export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
    export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
    docker run ${CUDA_SO} ${DEVICES} -it paddledev/paddle:0.10.0rc1-gpu


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

   docker run -d --name paddle-cpu-doc paddle:0.10.0rc1-cpu
   docker run -d --volumes-from paddle-cpu-doc -p 8088:80 nginx


Then we can direct our Web browser to the HTML version of source code
at http://localhost:8088/paddle/
