##################
如何贡献/修改文档
##################

PaddlePaddle的文档包括英文文档 ``doc`` 和中文文档 ``doc_cn`` 两个部分。文档都是通过 `cmake`_ 驱动 `sphinx`_ 编译生成，生成后的文档分别存储在编译目录的 ``doc`` 和 ``doc_cn`` 两个子目录下。
也可以利用PaddlePaddle 工具来编译文档，这个情况下所有的文件会存在整理过的的文件目录 .ppo_workspace/content 下

如何构建文档
============

PaddlePaddle的文档构建有三种方式。


使用PaddlePaddle.org工具
--------------
这个是目前推荐的使用方法。除了可以自动编译文档，也可以直接在网页预览文档。

文件工具是使用Docker，需要在系统里先安装好Docker工具包。Docker安装请参考Docker的官网。安装好Docker之后及可用以下命令启动工具

..  code-block:: bash

    mkdir paddlepaddle # Create paddlepaddle working directory
    cd paddlepaddle

    # Clone the content repositories
    git clone https://github.com/PaddlePaddle/Paddle.git
    git clone https://github.com/PaddlePaddle/book.git
    git clone https://github.com/PaddlePaddle/models.git
    git clone https://github.com/PaddlePaddle/Mobile.git

    # Please specify the working directory through -v
    docker run -it -p 8000:8000 -v `pwd`:/var/content paddlepaddle/paddlepaddle.org:latest

注意: PaddlePaddle.org 会在 -v (volume) 指定的内容存储库运行命令
之后再用网页连到http://localhost:8000就可以在网页上生成需要的文档
编译后的文件将被存储在工作目录 <paddlepaddle working directory>/.ppo_workspace/content。

如果不想使用 Docker，你还可以通过运行Django框架直接激活工具的服务器。使用下面的命令来运行它。

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

工具服务器将读取环境变量 CONTENT_DIR 搜索代码库。请指定的PaddlePaddle工作目录给环境变量 CONTENT_DIR。
之后再用网页连到http://localhost:8000就可以在网页上生成需要的文档。
编译后的文件将被存储在工作目录 <paddlepaddle working directory>/.ppo_workspace/content。

想了解更多PaddlePaddle.org工具的详细信息，可以 `点击这里 <https://github.com/PaddlePaddle/PaddlePaddle.org/blob/develop/README.cn.md>`_ 。

使用Docker构建
--------------

使用Docker构建PaddlePaddle的文档，需要在系统里先安装好Docker工具包。Docker安装请参考 `Docker的官网 <https://docs.docker.com/>`_ 。安装好Docker之后可以使用源码目录下的脚本构建文档，即

..  code-block:: bash

    cd TO_YOUR_PADDLE_CLONE_PATH
    cd paddle/scripts/tools/build_docs
    sh build_docs.sh

编译完成之后，会在当前目录生成两个子目录\: doc(英文文档目录)和 doc_cn(中文文档目录)。
打开浏览器访问对应目录下的index.html即可访问本地文档。

直接构建
--------

如果提示正确，可以执行以下命令编译生成文档，即

..  code-block:: bash

    cd TO_YOUR_PADDLE_CLONE_PATH
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Debug -DWITH_GPU=OFF -DWITH_MKL=OFF -DWITH_DOC=ON
    make gen_proto_py
    make paddle_docs paddle_docs_cn

编译完成之后，会在当前目录生成两个子目录\: doc(英文文档目录)和 doc_cn(中文文档目录)。
打开浏览器访问对应目录下的index.html即可访问本地文档。


如何书写文档
============

PaddlePaddle文档使用 `sphinx`_ 自动生成，用户可以参考sphinx教程进行书写。

如何更新www.paddlepaddle.org
============================

更新的文档以PR的形式提交到github中，提交方式参见 `贡献文档 <http://www.paddlepaddle.org/docs/develop/documentation/en/howto/dev/contribute_to_paddle_en.html>`_ 。
目前PaddlePaddle的develop分支的文档是自动触发更新的，用户可以分别查看最新的 `中文文档 <http://www.paddlepaddle.org/docs/develop/documentation/zh/getstarted/index_cn.html>`_ 和
`英文文档 <http://www.paddlepaddle.org/docs/develop/documentation/en/getstarted/index_en.html>`_ 。


..  _cmake: https://cmake.org/
..  _sphinx: http://www.sphinx-doc.org/en/1.4.8/
