###############################
如何贡献/修改PaddlePaddle的文档
###############################

PaddlePaddle的文档使用 `cmake`_ 驱动 `sphinx`_ 生成。公有两个文档，:code:`doc` 和 :code:`doc_cn` 。这两者会在 `cmake`_ 中进行编译，生成后的文档会存储在服务器的 :code:`doc` 和 :code:`doc_cn` 两个目录下。

下面分几个部分介绍一下PaddlePaddle文档的贡献方法。

如何书写PaddlePaddle的文档
==========================

TBD

如何构建PaddlePaddle的文档
==========================

构建PaddlePaddle文档，需要使用构建Paddle的全部环境。准备这个环境相对来说比较复杂，所以本文档提供两种方式构建PaddlePaddle的文档，即

* 使用Docker构建PaddlePaddle的文档
* 直接构建PaddlePaddle的文档。

并且，我们推荐使用Docker来构建PaddlePaddle的文档。


使用Docker构建PaddlePaddle的文档
--------------------------------

使用Docker构建PaddlePaddle的文档，首先要求在系统里安装好Docker工具包。安装Docker请参考 `Docker的官网 <https://docs.docker.com/>`_ 。

安装好Docker之后可以使用源码目录下的脚本构建文档，即

..	code-block:: bash

	cd TO_YOUR_PADDLE_CLONE_PATH
	cd paddle/scripts/tools/build_docs
	bash build_docs.sh

执行完这个脚本后，该目录下会生成两个目录，分别是\:

* doc 目录，英文文档地址
* doc_cn 目录，中文文档地址

打开浏览器访问对应目录下的index.html即可访问本地文档。

..	code-block:: bash

	open doc_cn/index.html


直接构建PaddlePaddle的文档
--------------------------

TBD


如何更新www.paddlepaddle.org文档
================================

TBD


..	_cmake: https://cmake.org/
..	_sphinx: http://www.sphinx-doc.org/en/1.4.8/