install and Compile
======================

.. _install_steps:

PaddlePaddle provides various methods of installation for many different users

Focus on Deep Learning Model Development
----------------------------------------

PaddlePaddle provides lots of packages of python wheel , that pip can install:

.. toctree::
	:maxdepth: 1

	pip_install_en.rst

This is the most convenient way of installation. Please choose the right installation package with machine configure and system.

Follow the Bottom Frame
------------------------

PaddlePaddle also supports installation using Docker. Please refer to the tutorial below:

.. toctree::
	:maxdepth: 1

	docker_install_en.rst

We recommend running PaddlePaddle in Docker. This method has the following advantages：

- Does not require installation of third-party dependencies. 
- Easy to share runtime environment. 

Lastly, users can also compile and install PaddlePaddle from source code. The instructions are below:

.. toctree::
    :maxdepth: 1

    build_from_source_en.rst

.. warning::

	One caveat with this approach is that developers will have to download, compile and install all third-party dependencies. Thus this process of installation is more time consuming.


FAQ
-----------

For any problems during installation, please refer to the page below for answers:

:ref:`常见问题解答 <install_faq>`

If the problem still persists, you are welcome to seek assistance from the PaddlePaddle community：

`创建issue <https://github.com/PaddlePaddle/Paddle/issues/new>`_
