Install and Build
=================

.. _quick_install:

Quick Install
----------------------

You can use pip to install PaddlePaddle using a single command, supports
CentOS 6 above, Ubuntu 14.04 above or MacOS 10.12, with Python 2.7 installed.
Simply run the following command to install:

  .. code-block:: bash

     pip install paddlepaddle

If you need to install GPU version, run:

  .. code-block:: bash

     pip install paddlepaddle-gpu


.. _install_steps:

Install Steps
++++++++

You can choose either pip or Docker to complete your install:

.. toctree::
   :maxdepth: 1

   pip_install_en.rst
   docker_install_en.rst


Build from Source
-----------------

..  warning::

    We recommend to directly install via above installation steps, you'll only need to build PaddlePaddle from source when you need a modifed binary.

..  toctree::
    :maxdepth: 1

    build_from_source_en.md
