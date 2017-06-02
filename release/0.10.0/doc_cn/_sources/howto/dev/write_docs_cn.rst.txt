##################
如何贡献/修改文档
##################

PaddlePaddle的文档包括英文文档 ``doc`` 和中文文档 ``doc_cn`` 两个部分。文档都是通过 `cmake`_ 驱动 `sphinx`_ 编译生成，生成后的文档分别存储在编译目录的 ``doc`` 和 ``doc_cn`` 两个子目录下。


如何构建PaddlePaddle的文档
==========================

PaddlePaddle的文档构建有直接构建和基于Docker构建两种方式。构建PaddlePaddle文档需要准备的环境相对较复杂，所以我们推荐使用基于Docker来构建PaddlePaddle的文档。


使用Docker构建PaddlePaddle的文档
--------------------------------

使用Docker构建PaddlePaddle的文档，需要在系统里先安装好Docker工具包。Docker安装请参考 `Docker的官网 <https://docs.docker.com/>`_ 。安装好Docker之后可以使用源码目录下的脚本构建文档，即

..	code-block:: bash

	cd TO_YOUR_PADDLE_CLONE_PATH
	cd paddle/scripts/tools/build_docs
	bash build_docs.sh

编译完成后，该目录下会生成如下两个子目录\:

* doc 英文文档目录
* doc_cn 中文文档目录

打开浏览器访问对应目录下的index.html即可访问本地文档。

..	code-block:: bash

	open doc_cn/index.html


直接构建PaddlePaddle的文档
--------------------------

TBD

如何书写PaddlePaddle的文档
==========================

TBD

如何更新www.paddlepaddle.org文档
================================

TBD


..	_cmake: https://cmake.org/
..	_sphinx: http://www.sphinx-doc.org/en/1.4.8/
