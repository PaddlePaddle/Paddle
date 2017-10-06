##################
如何贡献/修改文档
##################

PaddlePaddle的文档包括英文文档 ``doc`` 和中文文档 ``doc_cn`` 两个部分。文档都是通过 `cmake`_ 驱动 `sphinx`_ 编译生成，生成后的文档分别存储在编译目录的 ``doc`` 和 ``doc_cn`` 两个子目录下。


如何构建文档
============

PaddlePaddle的文档构建有两种方式。

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
    cmake .. -DCMAKE_BUILD_TYPE=Debug -DWITH_GPU=OFF -DWITH_MKLDNN=OFF -DWITH_MKLML=OFF -DWITH_DOC=ON
    make gen_proto_py
    make paddle_docs paddle_docs_cn

编译完成之后，会在当前目录生成两个子目录\: doc(英文文档目录)和 doc_cn(中文文档目录)。
打开浏览器访问对应目录下的index.html即可访问本地文档。


如何书写文档
============

PaddlePaddle文档使用 `sphinx`_ 自动生成，用户可以参考sphinx教程进行书写。

如何更新文档主题
================

PaddlePaddle文档主题在 `TO_YOUR_PADDLE_CLONE_PATH/doc_theme` 文件夹下，包含所有和前端网页设计相关的文件。

如何更新doc.paddlepaddle.org
============================

更新的文档以PR的形式提交到github中，提交方式参见 `贡献文档 <http://doc.paddlepaddle.org/develop/doc_cn/howto/dev/contribute_to_paddle_cn.html>`_ 。
目前PaddlePaddle的develop分支的文档是自动触发更新的，用户可以分别查看最新的 `中文文档 <http://doc.paddlepaddle.org/develop/doc_cn/>`_ 和
`英文文档 <http://doc.paddlepaddle.org/develop/doc/>`_ 。


..  _cmake: https://cmake.org/
..  _sphinx: http://www.sphinx-doc.org/en/1.4.8/
