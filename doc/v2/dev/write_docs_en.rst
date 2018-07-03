########################
Contribute Documentation
########################

PaddlePaddle's documentation includes both Chinese and English versions. The documentation is built using the ``cmake`` command to drive the ``sphinx`` compiler. The PaddlePaddle.org tool helps us to implement this compilation process and provides better preview results.

How to build Documentation
===========================

PaddlePaddle's documentation is built in two ways: using the PaddlePaddle.org tool and without using it. Both methods have their own advantages. The former facilitates previewing, while the latter facilitates debugging by the developer. We could choose to build the documentation with Docker or without it in each of the above ways.

We recommend using PaddlePaddle.org tool to build documentation.

Using PaddlePaddle.org tool
-----------------------------
This is the recommended method to build documentation, because it can automatically compile the documentation and preview the documentation directly in a web page. Note that, although you can preview the documentation in other ways, its style may not be consistent with the official website. Compiling with the PaddlePaddle.org tool produces a preview that will be consistent with the official website documentation style.

The PaddlePaddle.org tool can be used with Docker and Docker needs to be installed first. Please refer to `Docker's official website <https://docs.docker.com/>`_ on how to install Docker. After installing Docker, you may use the following commands to activate the tool

..  code-block:: bash

    mkdir paddlepaddle # Create paddlepaddle working directory
    cd paddlepaddle

    # Clone the content repositories. You may only clone the contents you need
    git clone https://github.com/PaddlePaddle/Paddle.git
    git clone https://github.com/PaddlePaddle/book.git
    git clone https://github.com/PaddlePaddle/models.git
    git clone https://github.com/PaddlePaddle/Mobile.git

    # Please specify the working directory through -v
    docker run -it -p 8000:8000 -v `pwd`:/var/content paddlepaddle/paddlepaddle.org:latest

Note: PaddlePaddle.org will read the content repos specified in the -v (volume) flag of the docker run commands
Use a web browser and navigate to http://localhost:8000. Click the buttons to compile the documentation.
The compiled documentations will be stored in <paddlepaddle working directory>/.ppo_workspace/content


If you don't wish to use Docker, you can also activate the tool through Django. Use the following the commands to set up

..  code-block:: bash

    mkdir paddlepaddle # Create paddlepaddle working directory
    cd paddlepaddle

    # Clone the content repositories and PaddlePaddle.org
    git clone https://github.com/PaddlePaddle/Paddle.git
    git clone https://github.com/PaddlePaddle/book.git
    git clone https://github.com/PaddlePaddle/models.git
    git clone https://github.com/PaddlePaddle/Mobile.git
    git clone https://github.com/PaddlePaddle/PaddlePaddle.org.git

    # Please specify the PaddlePaddle working directory. In the current setting, it should be pwd
    export CONTENT_DIR=<path_to_paddlepaddle_working_directory>
    export ENV=''
    cd PaddlePaddle.org/portal/
    pip install -r requirements.txt
    python manage.py runserver

Specify the PaddlePaddle working directory for the environment variable CONTENT_DIR so that the tool could find where the working directory is.

Use a web browser and navigate to http://localhost:8000. Click the buttons to compile the documentation
The compiled documentations will be stored in <paddlepaddle working directory>/.ppo_workspace/content

Please `click here <https://github.com/PaddlePaddle/PaddlePaddle.org/blob/develop/README.md>`_ for more information about the PaddlePaddle.org tool.


Manually Building the Documentation
-------------------------------------

Build PaddlePaddle's documentation with Docker，you need to install Docker first. Please refer to `Docker's official website <https://docs.docker.com/>`_ on how to install Docker. This method is quite similar to ` Build From Sources <http://paddlepaddle.org/docs/develop/documentation/en/build_and_install/build_from_source_en.html>`_ , by constructing, from source code, a docker image that can be used to build PaddlePaddle documentation. Enter the Docker container and use the script ``build.sh`` in the source directory to build the PaddlePaddle documentation. The specific steps are as follows:

.. code-block:: bash

   git clone https://github.com/PaddlePaddle/Paddle.git
   cd Paddle

   # Construct a docker image from source code
   docker build -t paddle:dev .
   docker run -it -v $PWD:/paddle -e "WITH_GPU=OFF" -e "WITH_TESTING=OFF" -e "WITH_DOC=ON" paddle:dev /bin/bash

   # Use build.sh to build PaddlePaddle documentation
   bash -x /paddle/paddle/scripts/docker/build.sh

Note: The above commands maps the current directory (source root directory) to the :code:`/paddle` directory in the container.

After compiling, there should be two generated directories: ``doc/v2`` and ``doc/fluid``, where three subdirectories ``cn/html/``, ``en/html`` and ``api/en/html`` are generated. Please enter these directories respectively and execute the following commands:

.. code-block:: bash

   python -m SimpleHTTPServer 8088

Use a web browser and navigate to http://localhost:8000, you could see the compiled  ``v2`` 's and ``fluid`` 's Chinese/English documents page and English APIs page.

If you do not wish to use Docker, you can also use the following commands to directly build the PaddlePaddle documentation.

.. code-block:: bash


   git clone https://github.com/PaddlePaddle/Paddle.git
   cd Paddle
   mkdir -p build
   cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=OFF -DWITH_MKL=OFF -DWITH_DOC=ON

   # If you only need to build documents, use the following commands
   make -j $processors paddle_docs

   # If you only need to build APIs, use the following commands
   make -j $processors paddle_apis

$processors indicates that as many processes as the CPU cores are started to compile in parallel. It should be set according to the number of CPU cores of your machine.

After compiling, there also should be two generated directories: ``doc/v2`` and ``doc/fluid`` . If you chose to build documents, two subdirectories ``cn/html/`` and ``en/html``  will be generated in both two directories. If you chose to build APIs，a subdirectory ``api/en/html`` will be generated. Please enter these directories respectively and execute the following commands:

.. code-block:: bash

   python -m SimpleHTTPServer 8088

Use a web browser and navigate to http://localhost:8000, you could see the compiled  ``v2`` 's and ``fluid`` 's Chinese/English documents page and English APIs page. The following figure is an example of the built ``v2`` 's English documents home page. Note that due to the sphinx's original theme used in the example, the style of the page is not consistent with the official website, but this does not affect the developer's debugging.

..  image:: src/doc_en.png
    :align: center
    :scale: 60 %

How to write Documentation
===========================

PaddlePaddle uses `sphinx`_ to compile documentation，Please check sphinx official website for more detail.

How to update www.paddlepaddle.org
===================================

Please create PRs and submit them to github, please check `Contribute Code <http://www.paddlepaddle.org/docs/develop/documentation/en/howto/dev/contribute_to_paddle_en.html>`_ 。
PaddlePaddle develop branch will update the documentation once the PR is merged. User may check latest `Chinese Docs <http://www.paddlepaddle.org/docs/develop/documentation/zh/getstarted/index_cn.html>`_ and
`English Docs <http://www.paddlepaddle.org/docs/develop/documentation/en/getstarted/index_en.html>`_ 。

..  _cmake: https://cmake.org/
..  _sphinx: http://www.sphinx-doc.org/en/1.4.8/
