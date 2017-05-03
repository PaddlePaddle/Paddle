PaddlePaddle in Docker Containers
=================================

Docker container is currently the only officially-supported way to
running PaddlePaddle.  This is reasonable as Docker now runs on all
major operating systems including Linux, Mac OS X, and Windows.
Please be aware that you will need to change `Dockers settings
<https://github.com/PaddlePaddle/Paddle/issues/627>`_ to make full use
of your hardware resource on Mac OS X and Windows.

Working With Docker
-------------------

Docker is simple as long as we understand a few basic concepts:

- *image*: A Docker image is a pack of software. It could contain one or more programs and all their dependencies. For example, the PaddlePaddle's Docker image includes pre-built PaddlePaddle and Python and many Python packages. We can run a Docker image directly, other than installing all these software. We can type

  .. code-block:: bash

     docker images

  to list all images in the system. We can also run

  .. code-block:: bash
		  
     docker pull paddlepaddle/paddle:0.10.0rc2

  to download a Docker image, paddlepaddle/paddle in this example,
  from Dockerhub.com.

- *container*: considering a Docker image a program, a container is a
  "process" that runs the image. Indeed, a container is exactly an
  operating system process, but with a virtualized filesystem, network
  port space, and other virtualized environment. We can type

  .. code-block:: bash

     docker run paddlepaddle/paddle:0.10.0rc2

  to start a container to run a Docker image, paddlepaddle/paddle in this example.

- By default docker container have an isolated file system namespace,
  we can not see the files in the host file system. By using *volume*,
  mounted files in host will be visible inside docker container.
  Following command will mount current dirctory into /data inside
  docker container, run docker container from debian image with
  command :code:`ls /data`.

  .. code-block:: bash

     docker run --rm -v $(pwd):/data debian ls /data

Usage of CPU-only and GPU Images
----------------------------------

We package PaddlePaddle's compile environment into a Docker image,
called the develop image, it contains all compiling tools that
PaddlePaddle needs. We package compiled PaddlePaddle program into a
Docker image as well, called the production image, it contains all
runtime environment that running PaddlePaddle needs. For each version
of PaddlePaddle, we release both of them. Production image includes
CPU-only version and a CUDA GPU version and their no-AVX versions.

We put the docker images on `dockerhub.com
<https://hub.docker.com/r/paddledev/paddle/>`_. You can find the
latest versions under "tags" tab at dockerhub.com. If you are in
China, you can use our Docker image registry mirror to speed up the
download process. To use it, please replace all paddlepaddle/paddle in
the commands to docker.paddlepaddle.org/paddle.

1. Production images, this image might have multiple variants:

   - GPU/AVX：:code:`paddlepaddle/paddle:<version>-gpu`
   - GPU/no-AVX：:code:`paddlepaddle/paddle:<version>-gpu-noavx`
   - CPU/AVX：:code:`paddlepaddle/paddle:<version>`
   - CPU/no-AVX：:code:`paddlepaddle/paddle:<version>-noavx`

   Please be aware that the CPU-only and the GPU images both use the
   AVX instruction set, but old computers produced before 2008 do not
   support AVX.  The following command checks if your Linux computer
   supports AVX:

   .. code-block:: bash

      if cat /proc/cpuinfo | grep -i avx; then echo Yes; else echo No; fi

   
   To run the CPU-only image as an interactive container:

   .. code-block:: bash

      docker run -it --rm paddlepaddle/paddle:0.10.0rc2 /bin/bash

   Above method work with the GPU image too -- the recommended way is
   using `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_.

   Please install nvidia-docker first following this `tutorial
   <https://github.com/NVIDIA/nvidia-docker#quick-start>`_.

   Now you can run a GPU image:

   .. code-block:: bash

      nvidia-docker run -it --rm paddlepaddle/paddle:0.10.0rc2-gpu /bin/bash

2. development image :code:`paddlepaddle/paddle:<version>-dev`

   This image has packed related develop tools and runtime
   environment. Users and developers can use this image instead of
   their own local computer to accomplish development, build,
   releasing, document writing etc. While different version of paddle
   may depends on different version of libraries and tools, if you
   want to setup a local environment, you must pay attention to the
   versions.  The development image contains:
   
   - gcc/clang
   - nvcc
   - Python
   - sphinx
   - woboq
   - sshd
     
   Many developers use servers with GPUs, they can use ssh to login to
   the server and run :code:`docker exec` to enter the docker
   container and start their work.  Also they can start a development
   docker image with SSHD service, so they can login to the container
   and start work.


Train Model Using Python API
----------------------------

Our official docker image provides a runtime for PaddlePaddle
programs. The typical workflow will be as follows:

Create a directory as workspace:

.. code-block:: bash

   mkdir ~/workspace

Edit a PaddlePaddle python program using your favourite editor

.. code-block:: bash

   emacs ~/workspace/example.py

Run the program using docker:

.. code-block:: bash

   docker run --rm -v ~/workspace:/workspace paddlepaddle/paddle:0.10.0rc2 python /workspace/example.py

Or if you are using GPU for training:

.. code-block:: bash

   nvidia-docker run --rm -v ~/workspace:/workspace paddlepaddle/paddle:0.10.0rc2-gpu python /workspace/example.py

Above commands will start a docker container by running :code:`python
/workspace/example.py`. It will stop once :code:`python
/workspace/example.py` finishes.

Another way is to tell docker to start a :code:`/bin/bash` session and
run PaddlePaddle program interactively:

.. code-block:: bash

   docker run -it -v ~/workspace:/workspace paddlepaddle/paddle:0.10.0rc2 /bin/bash
   # now we are inside docker container
   cd /workspace
   python example.py

Running with GPU is identical:

.. code-block:: bash

   nvidia-docker run -it -v ~/workspace:/workspace paddlepaddle/paddle:0.10.0rc2-gpu /bin/bash
   # now we are inside docker container
   cd /workspace
   python example.py


Develop PaddlePaddle or Train Model Using C++ API
---------------------------------------------------

We will be using PaddlePaddle development image since it contains all
compiling tools and dependencies.

1. Build PaddlePaddle develop image

   Use following command to build PaddlePaddle develop image:

   .. code-block:: bash

      git clone https://github.com/PaddlePaddle/Paddle.git && cd Paddle
      docker build -t paddle:dev .

2. Build PaddlePaddle production image

   There are two steps for building production image, the first step is to run:

   .. code-block:: bash

      docker run -v $(pwd):/paddle -e "WITH_GPU=OFF" -e "WITH_AVX=OFF" -e "WITH_TEST=ON" paddle:dev

   The above command will compile PaddlePaddle and create a Dockerfile for building production image. All the generated files are in the build directory. "WITH_GPU" controls if the generated production image supports GPU. "WITH_AVX" controls if the generated production image supports AVX. "WITH_TEST" controls if the unit test will be generated.

   The second step is to run:

   .. code-block:: bash

      docker build -t paddle:prod -f build/Dockerfile ./build

   The above command will generate the production image by copying the compiled PaddlePaddle program into the image.

3. Run unit test

   Following command will run unit test:

   .. code-block:: bash
      
      docker run -it -v $(pwd):/paddle paddle:dev bash -c "cd /paddle/build && ctest"

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
