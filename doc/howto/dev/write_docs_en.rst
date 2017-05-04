###############
Build Documents
###############

Document files of PaddlePaddle are in sub-directory :code:`doc`.  Source files are in `RST <http://docutils.sourceforge.net/docs/user/rst/quickref.html>`_ format.  We can build the document by letting `cmake<https://cmake.org/>`_ invoke `sphinx <http://www.sphinx-doc.org/en/1.4.8/>`_  to convert RST files into HTML files.


How to Build Documents
======================

To save the time of installing building tools, we provide a Docker image.


使用Docker构建PaddlePaddle的文档
--------------------------------

使用Docker构建PaddlePaddle的文档，需要在系统里先安装好Docker工具包。Docker安装请参考 `Docker的官网 <https://docs.docker.com/>`_ 。安装好Docker之后可以使用源码目录下的脚本构建文档，即

..  code-block:: bash

    cd TO_YOUR_PADDLE_CLONE_PATH
    cd paddle/scripts/tools/build_docs
    bash build_docs.sh with_docker

编译完成后，会在当前目录生成两个子目录\:

* doc 英文文档目录
* doc_cn 中文文档目录

打开浏览器访问对应目录下的index.html即可访问本地文档。



直接构建PaddlePaddle的文档
--------------------------

因为PaddlePaddle的v2 api文档生成过程依赖于py_paddle Python包，用户需要首先确认py_paddle包已经安装。

..  code-block:: bash

    python -c "import py_paddle"

如果提示错误，那么用户需要在本地编译安装PaddlePaddle，请参考 `源码编译文档 <http://www.paddlepaddle.org/develop/doc/getstarted/build_and_install/build_from_source_en.html>`_ 。
注意，用户在首次编译安装PaddlePaddle时，请将WITH_DOC选项关闭。在编译安装正确之后，请再次确认py_paddle包已经安装，即可进行下一步操作。

如果提示正确，可以执行以下命令编译生成文档，即

..  code-block:: bash

    cd TO_YOUR_PADDLE_CLONE_PATH
    cd paddle/scripts/tools/build_docs
    bash build_docs.sh local

编译完成之后，会在当前目录生成两个子目录\:

* doc 英文文档目录
* doc_cn 中文文档目录

打开浏览器访问对应目录下的index.html即可访问本地文档。


如何书写PaddlePaddle的文档
==========================

PaddlePaddle文档使用 `sphinx`_ 自动生成，用户可以参考sphinx教程进行书写。

如何更新www.paddlepaddle.org文档
================================

开发者给PaddlePaddle代码增加的注释以PR的形式提交到github中，提交方式可参见 `贡献文档 <http://paddlepaddle.org/develop/doc_cn/howto/dev/contribute_to_paddle_cn.html>`_ 。
目前PaddlePaddle的develop分支的文档是自动触发更新的，用户可以分别查看最新的 `中文文档 <http://www.paddlepaddle.org/develop/doc_cn/>`_ 和
`英文文档 <http://www.paddlepaddle.org/develop/doc/>`_ 。



..  _cmake: https://cmake.org/
..  _sphinx: http://www.sphinx-doc.org/en/1.4.8/
