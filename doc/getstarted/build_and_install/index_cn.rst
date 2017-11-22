安装与编译
==========

.. _quick_install:

快速安装
++++++++

PaddlePaddle支持使用pip快速安装，目前支持CentOS 6以上, Ubuntu 14.04以及MacOS 10.12，并安装有Python2.7。
执行下面的命令完成快速安装：

  .. code-block:: bash

     pip install paddlepaddle

如果需要安装支持GPU的版本，需要执行：

  .. code-block:: bash

     pip install paddlepaddle-gpu

.. _install_steps:

安装流程
++++++++

PaddlePaddle提供pip和Docker的安装方式：

.. toctree::
   :maxdepth: 1

   pip_install_cn.rst
   docker_install_cn.rst


编译流程
++++++++

..  warning::

    建议直接使用上述安装流程，方便快速安装。只有在遇到需要独立定制的二进制时才需要编译。

..  toctree::
    :maxdepth: 1

    build_from_source_cn.rst
