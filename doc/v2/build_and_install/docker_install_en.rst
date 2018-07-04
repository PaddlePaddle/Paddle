Run in Docker Containers
=================================

Run PaddlePaddle in Docker container so that you don't need to care about
runtime dependencies, also you can run under Windows system. You can get
tutorials at `here <https://docs.docker.com/get-started/>`_ .

If you are using Windows, please refer to
`this <https://docs.docker.com/toolbox/toolbox_install_windows/>`_
tutorial to start running docker under windows.

After you've read above tutorials you may proceed the following steps.

.. _docker_pull:

Pull PaddlePaddle Docker Image
------------------------------

Run the following command to download the latest Docker images, the version is cpu_avx_mkl:

  .. code-block:: bash

     docker pull paddlepaddle/paddle

For users in China, we provide a faster mirror:

  .. code-block:: bash

     docker pull docker.paddlepaddlehub.com/paddle

Download GPU version (cuda8.0_cudnn5_avx_mkl) images:

  .. code-block:: bash

     docker pull paddlepaddle/paddle:latest-gpu
     docker pull docker.paddlepaddlehub.com/paddle:latest-gpu

Choose between different BLAS version:

  .. code-block:: bash

     # image using MKL by default
     docker pull paddlepaddle/paddle
     # image using OpenBLAS
     docker pull paddlepaddle/paddle:latest-openblas


If you want to use legacy versions, choose a tag from
`DockerHub <https://hub.docker.com/r/paddlepaddle/paddle/tags/>`_
and run:

  .. code-block:: bash

     docker pull paddlepaddle/paddle:[tag]
     # i.e.
     docker pull docker.paddlepaddlehub.com/paddle:0.11.0-gpu

.. _docker_run:

Launch your training program in Docker
--------------------------------------

Assume that you have already written a PaddlePaddle program
named :code:`train.py` under directory :code:`/home/work` (refer to 
`PaddlePaddleBook <http://www.paddlepaddle.org/docs/develop/book/01.fit_a_line/index.cn.html>`_ 
for more samples), then run the following command:

  .. code-block:: bash

     cd /home/work
     docker run -it -v $PWD:/work paddlepaddle/paddle /work/train.py

In the above command, :code:`-it` means run the container interactively;
:code:`-v $PWD:/work` means mount the current directory ($PWD will expand
to current absolute path in Linux) under :code:`/work` in the container.
:code:`paddlepaddle/paddle` to specify image to use; finnally
:code:`/work/train.py` is the command to run inside docker.

Also, you can go into the container shell, run or debug your code
interactively:

  .. code-block:: bash

     docker run -it -v $PWD:/work paddlepaddle/paddle /bin/bash
     cd /work
     python train.py

**NOTE: We did not install vim in the default docker image to reduce the image size, you can run** :code:`apt-get install -y vim` **to install it if you need to edit python files.**

.. _docker_run_book:

PaddlePaddle Book
------------------

You can create a container serving PaddlePaddle Book using Jupyter Notebook in
one minute using Docker. PaddlePaddle Book is an interactive Jupyter Notebook
for users and developers.If you want to
dig deeper into deep learning, PaddlePaddle Book definitely is your best choice.

We provide a packaged book image, simply issue the command:

  .. code-block:: bash

     docker run -p 8888:8888 paddlepaddle/book

For users in China, we provide a faster mirror:

  .. code-block:: bash

    docker run -p 8888:8888 docker.paddlepaddlehub.com/book

Then, you would back and paste the address into the local browser:

  .. code-block:: text

     http://localhost:8888/

That's all. Enjoy your journey!

.. _docker_run_gpu:

Train with Docker with GPU
------------------------------

We recommend using
`nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_
to run GPU training jobs. Please ensure you have latest
GPU driver installed before move on.

  .. code-block:: bash

     nvidia-docker run -it -v $PWD:/work paddlepaddle/paddle:latest-gpu /bin/bash

**NOTE: If you don't have nvidia-docker installed, try the following method to mount CUDA libs and devices into the container.**

  .. code-block:: bash

     export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
     export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
     docker run ${CUDA_SO} ${DEVICES} -it paddlepaddle/paddle:latest-gpu

**About AVX:**

AVX is a kind of CPU instruction can accelerate PaddlePaddle's calculations.
The latest PaddlePaddle Docker image turns AVX on by default, so, if your
computer doesn't support AVX, you'll probably need to
`build <./build_from_source_en.html>`_ with :code:`WITH_AVX=OFF`.

The following command will tell you whether your computer supports AVX.

   .. code-block:: bash

      if cat /proc/cpuinfo | grep -i avx; then echo Yes; else echo No; fi
