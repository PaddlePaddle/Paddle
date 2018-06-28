安装与编译
==========

.. _install_steps:

PaddlePaddle针对不同的用户群体提供了多种安装方式。

专注深度学习模型开发
--------------------

PaddlePaddle提供了多种python wheel包，可通过pip一键安装：

.. toctree::
	:maxdepth: 1

	pip_install_cn.rst

这是最便捷的安装方式，请根据机器配置和系统选择对应的安装包。

关注底层框架
-------------

PaddlePaddle提供了基于Docker的安装方式，请参照以下教程：

.. toctree::
	:maxdepth: 1

	docker_install_cn.rst

我们推荐在Docker中运行PaddlePaddle，该方式具有以下优势：

- 无需单独安装第三方依赖
- 方便分享运行时环境，易于问题的复现

对于有定制化二进制文件需求的用户，我们同样提供了从源码编译安装PaddlePaddle的方法：

.. toctree::
    :maxdepth: 1

    build_from_source_cn.rst

.. warning::

	需要提醒的是，这种安装方式会涉及到一些第三方库的下载、编译及安装，整个安装过程耗时较长。


常见问题汇总
--------------

如果在安装过程中遇到了问题，请先尝试在下面的页面寻找答案：

:ref:`常见问题解答 <install_faq>`

如果问题没有得到解决，欢迎向PaddlePaddle社区反馈问题：

`创建issue <https://github.com/PaddlePaddle/Paddle/issues/new>`_
