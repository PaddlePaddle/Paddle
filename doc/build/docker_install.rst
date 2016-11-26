Using and Building Docker Images
================================

We release PaddlePaddle in the form of `Docker <https://www.docker.com/>`_ images on `dockerhub.com <https://hub.docker.com/r/paddledev/paddle/>`_.   Running as Docker containers is currently the only officially-supported way to running PaddlePaddle.

Run Docker images
-----------------

For each version of PaddlePaddle, we release 4 variants of Docker images:

+-----------------+-------------+-------+
|                 |   CPU AVX   |  GPU  |
+=================+=============+=======+
|       cpu       |   yes       |  no   |
+-----------------+-------------+-------+
|    cpu-noavx    |   no        |  no   |
+-----------------+-------------+-------+
|       gpu       |   yes       |  yes  |
+-----------------+-------------+-------+
|    gpu-noavx    |   no        |  yes  |
+-----------------+-------------+-------+

The following command line detects if your CPU supports :code:`AVX`.

..  code-block:: bash

    if cat /proc/cpuinfo | grep -q avx ; then echo "Support AVX"; else echo "Not support AVX"; fi


Once we determine the proper variant, we can cope with the Docker image tag name by appending the version number.  For example, the following command runs the AVX-enabled image of the most recent version:

..  code-block:: bash

    docker run -it --rm paddledev/paddle:cpu-latest /bin/bash

To run a GPU-enabled image, you need to install CUDA and let Docker knows about it:

..  code-block:: bash

    export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
    export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
    docker run ${CUDA_SO} ${DEVICES} -it paddledev/paddle:gpu-latest

The default entry point of all our Docker images starts the OpenSSH server.  To run PaddlePaddle and to expose OpenSSH port to 2202 on the host computer:

..  code-block:: bash

    docker run -d -p 2202:22 paddledev/paddle:cpu-latest

Then we can login to the container using username :code:`root` and password :code:`root`:

..  code-block:: bash

    ssh -p 2202 root@localhost


Build Docker images
-------------------

Developers might want to build Docker images from their local commit or from a tagged version.  Suppose that your local repo is at :code:`~/work/Paddle`, the following steps builds a cpu variant from your current work:

.. code-block:: bash

  cd ~/Paddle
  ./paddle/scripts/docker/generates.sh # Use m4 to generate Dockerfiles for each variant.
  docker build -t paddle:latest -f ./paddle/scripts/docker/Dockerfile.cpu

As a release engineer, you might want to build Docker images for a certain version and publish them to dockerhub.com.  You can do this by switching to the right Git tag, or create a new tag, before running `docker build`.  For example, the following commands build Docker images for v0.9.0:

.. code-block:: bash

   cd ~/Paddle
   git checkout tags/v0.9.0
   ./paddle/scripts/docker/generates.sh # Use m4 to generate Dockerfiles for each variant.
   docker build -t paddle:cpu-v0.9.0 -f ./paddle/scripts/docker/Dockerfile.cpu
