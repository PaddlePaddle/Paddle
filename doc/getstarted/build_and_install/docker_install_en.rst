PaddlePaddle in Docker Containers
=================================

Docker container is currently the only officially-supported way to
running PaddlePaddle.  This is reasonable as Docker now runs on all
major operating systems including Linux, Mac OS X, and Windows.
Please be aware that you will need to change `Dockers settings
<https://github.com/PaddlePaddle/Paddle/issues/627>`_ to make full use
of your hardware resource on Mac OS X and Windows.


Usage of CPU-only and GPU Images
----------------------------------

For each version of PaddlePaddle, we release 2 types of Docker images: development
image and production image. Production image includes CPU-only version and a CUDA
GPU version and their no-AVX versions. We put the docker images on
`dockerhub.com <https://hub.docker.com/r/paddledev/paddle/>`_. You can find the
latest versions under "tags" tab at dockerhub.com.
1. development image :code:`paddlepaddle/paddle:<version>-dev`

    This image has packed related develop tools and runtime environment. Users and
    developers can use this image instead of their own local computer to accomplish
    development, build, releasing, document writing etc. While different version of
    paddle may depends on different version of libraries and tools, if you want to
    setup a local environment, you must pay attention to the versions.
    The development image contains:
    - gcc/clang
    - nvcc
    - Python
    - sphinx
    - woboq
    - sshd
    Many developers use servers with GPUs, they can use ssh to login to the server
    and run :code:`docker exec` to enter the docker container and start their work.
    Also they can start a development docker image with SSHD service, so they can login to
    the container and start work.

    To run the CPU-only image as an interactive container:

    .. code-block:: bash

        docker run -it --rm paddledev/paddle:<version> /bin/bash

    or, we can run it as a daemon container

    .. code-block:: bash

        docker run -d -p 2202:22 -p 8888:8888 paddledev/paddle:<version>

    and SSH to this container using password :code:`root`:

    .. code-block:: bash

        ssh -p 2202 root@localhost

    An advantage of using SSH is that we can connect to PaddlePaddle from
    more than one terminals.  For example, one terminal running vi and
    another one running Python interpreter.  Another advantage is that we
    can run the PaddlePaddle container on a remote server and SSH to it
    from a laptop.


2. Production images, this image might have multiple variants:
    - GPU/AVX：:code:`paddlepaddle/paddle:<version>-gpu`
    - GPU/no-AVX：:code:`paddlepaddle/paddle:<version>-gpu-noavx`
    - CPU/AVX：:code:`paddlepaddle/paddle:<version>`
    - CPU/no-AVX：:code:`paddlepaddle/paddle:<version>-noavx`

    Please be aware that the CPU-only and the GPU images both use the AVX
    instruction set, but old computers produced before 2008 do not support
    AVX.  The following command checks if your Linux computer supports
    AVX:

    .. code-block:: bash

       if cat /proc/cpuinfo | grep -i avx; then echo Yes; else echo No; fi


       If it doesn't, we will use the non-AVX images.

    Notice please don't forget
    to install CUDA driver and let Docker knows about it:

    .. code-block:: bash

        export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
        export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
        docker run ${CUDA_SO} ${DEVICES} -it paddledev/paddle:<version>-gpu


3. Use production image to release you AI application
    Suppose that we have a simple application program in :code:`a.py`, we can test and run it using the production image:

    ```bash
    docker run -it -v $PWD:/work paddle /work/a.py
    ```

    But this works only if all dependencies of :code:`a.py` are in the production image. If this is not the case, we need to build a new Docker image from the production image and with more dependencies installs.


PaddlePaddle Book
------------------

The Jupyter Notebook is an open-source web application that allows
you to create and share documents that contain live code, equations,
visualizations and explanatory text in a single browser.

PaddlePaddle Book is an interactive Jupyter Notebook for users and developers.
We already exposed port 8888 for this book. If you want to
dig deeper into deep learning, PaddlePaddle Book definitely is your best choice.

We provide a packaged book image, simply issue the command:

.. code-block:: bash

    docker run -p 8888:8888 paddlepaddle/book

Then, you would back and paste the address into the local browser:

.. code-block:: text

    http://localhost:8888/

That's all. Enjoy your journey!

Development Using Docker
------------------------

Developers can work on PaddlePaddle using Docker.  This allows
developers to work on different platforms -- Linux, Mac OS X, and
Windows -- in a consistent way.

1. Build the Development Docker Image

   .. code-block:: bash

      git clone --recursive https://github.com/PaddlePaddle/Paddle
      cd Paddle
      docker build -t paddle:dev .

   Note that by default :code:`docker build` wouldn't import source
   tree into the image and build it.  If we want to do that, we need docker the
   development docker image and then run the following command:

   .. code-block:: bash

      docker run -v $PWD:/paddle -e "WITH_GPU=OFF" -e "WITH_AVX=ON" -e "TEST=OFF" paddle:dev


2. Run the Development Environment

   Once we got the image :code:`paddle:dev`, we can use it to develop
   Paddle by mounting the local source code tree into a container that
   runs the image:

   .. code-block:: bash

      docker run -d -p 2202:22 -p 8888:8888 -v $PWD:/paddle paddle:dev sshd

   This runs a container of the development environment Docker image
   with the local source tree mounted to :code:`/paddle` of the
   container.

   The above :code:`docker run` commands actually starts
   an SSHD server listening on port 2202.  This allows us to log into
   this container with:

   .. code-block:: bash

      ssh root@localhost -p 2202

   Usually, I run above commands on my Mac.  I can also run them on a
   GPU server :code:`xxx.yyy.zzz.www` and ssh from my Mac to it:

   .. code-block:: bash

      my-mac$ ssh root@xxx.yyy.zzz.www -p 2202

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

   docker run -d --name paddle-cpu-doc paddle:<version>
   docker run -d --volumes-from paddle-cpu-doc -p 8088:80 nginx


Then we can direct our Web browser to the HTML version of source code
at http://localhost:8088/paddle/
