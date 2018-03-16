########################
Contribute Documentation
########################

PaddlePaddle's documentation includes both Chinese and English versions. The documentation is built using the ``cmake`` command to drive the ``sphinx`` compiler. The PaddlePaddle.org tool helps us to implement this compilation process and provides better preview results.

How to build Documentation
===========================

PaddlePaddle's documentation is built in two ways: using the PaddlePaddle.org tool and not using the PaddlePaddle.org tool. Both methods have their own advantages. The former facilitates previewing, while the latter facilitates debugging by the developer. And we could chose to build the documentation with docker or without docker in each way.

We recommend using PaddlePaddle.org tool to build documentation.

With PaddlePaddle.org tool
---------------------------
This is the recommended method to build documentation, because it can automatically compile the documentation and preview the documentation directly in the web page, it should be noted that, in other ways, although you can preview the documentation, but the style of the documentation and the official website documentation is inconsistent, while Compiling with the PaddlePaddle.org tool produces a preview that is consistent with the official website documentation style.

The PaddlePaddle.org tool can be used with Docker and Docker needs to be installed first. Please refer to the `Docker's official website <https://docs.docker.com/>`_ on how to install Docker. After installing Docker, you may use the following command to activate the tool

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

Note: PaddlePaddle.org will read the content repos specified in the -v (volume) flag of the docker run command
Use a web browser and navigate to http://localhost:8000, click the buttons to compile the documentation.
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

Use a web browser and navigate to http://localhost:8000, click the buttons to compile the documentation
The compiled documentations will be stored in <paddlepaddle working directory>/.ppo_workspace/content

Please `click here <https://github.com/PaddlePaddle/PaddlePaddle.org/blob/develop/README.md>`_ to get more informations about PaddlePaddle.org tool。


Without PaddlePaddle.org tool
-------------------------------

Build PaddlePaddle's documentation with Docker，you need to install Docker first. Please refer to the `Docker's official website <https://docs.docker.com/>`_ on how to install Docker. After Docker is installed, you could use the scripts in the source directory to build the documentation.

[TBD]

If you do not wish to use Docker, you can also use the following command to directly build the PaddlePaddle documentation.

.. code-block:: bash

   mkdir paddle
   cd paddle
   git clone https://github.com/PaddlePaddle/Paddle.git
   mkdir -p build
   cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=OFF -DWITH_MKL=OFF -DWITH_DOC=ON

   # If you only need to build documents, use the following command
   make -j $processors gen_proto_py
   make -j $processors paddle_docs paddle_docs_cn

   # If you only need to build APIs, use the following command
   make -j $processors gen_proto_py framework_py_proto
   make -j $processors copy_paddle_pybind
   make -j $processors paddle_api_docs

$processors indicates that as many processes as the CPU cores are started to compile in parallel. It should be set according to the number of CPU cores of your machine.

After the compilation is complete, enter the ``doc/v2`` directory. If you chose to build documents, it will generate ``cn/html/`` and ``en/html`` subdirectories under this directory. If you chose to build APIs，it will generate``api/en/html`` subdirectory. Please enter these directories respectively and execute the following command:

.. code-block:: bash

   python -m SimpleHTTPServer 8088

Use a web browser and navigate to http://localhost:8000, you could see the compiled Chinese/English documents page and the English APIs page. The following figure is an example of the built English documents home page. Note that due to the sphinx's original theme used in the example, the style of the page is not consistent with the official website, but this does not affect the developer's debugging.

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
