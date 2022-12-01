
--------------------------------------------------------------------------------

[![Build Status](https://travis-ci.org/PaddlePaddle/Paddle.svg?branch=develop)](https://travis-ci.org/PaddlePaddle/Paddle)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](http://www.paddlepaddle.org.cn/documentation/docs/en/1.8/beginners_guide/index_en.html)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](http://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/beginners_guide/index_cn.html)
[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle.svg)](https://github.com/PaddlePaddle/Paddle/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

Welcome to the PaddlePaddle GitHub.

PaddlePaddle, as the only independent R&D deep learning platform in China, has been officially open-sourced to professional communities since 2016. It is an industrial platform with advanced technologies and rich features that cover core deep learning frameworks, basic model libraries, end-to-end development kits, tools & components as well as service platforms.
PaddlePaddle is originated from industrial practices with dedication and commitments to industrialization. It has been widely adopted by a wide range of sectors including manufacturing, agriculture, enterprise service, and so on while serving more than 2.3 million developers. With such advantages, PaddlePaddle has helped an increasing number of partners commercialize AI.



## Installation

We provide users with four installation methods ,which are pip, conda, docker and install with source code.

### PIP Installation

#### PREQUISTIES

##### On Windows:

- **Windows 7/8/10 Pro/Enterprise (64bit)**
  - **GPU version support CUDA 10.2/11.2/11.6/11.7**
  - **Only supports single card**
- **Python version 2.7.15+/3.5.1+/3.6/3.7/3.8/3.9/3.10 (64 bit)**
- **pip version 9.0.1+ (64 bit)**

##### On Linux:

- **Linux Version (64 bit)**
  - **CentOS 7 (GPUVersion Supports CUDA 10.2/11.2/11.6/11.7)**
  - **Ubuntu 16.04/18.04/20.04/22.04 (GPUVersion Supports CUDA 10.2/11.2/11.6/11.7)**
- **Python Version: 2.7.15+/3.5.1+/3.6/3.7/3.8/3.9/3.10 (64 bit)**
- **pip or pip3 Version 20.2.2+ (64 bit)**

##### On macOS:

- **MacOS version 10.11/10.12/10.13/10.14 (64 bit) (not support GPU version yet)**

- **Python version 2.7.15+/3.5.1+/3.6/3.7/3.8/3.9/3.10 (64 bit)**

- **pip or pip3 version 9.0.1+ (64 bit)**



#### Commands to install

###### cpu:

```pip install paddlepaddle```

###### gpu:

```pip install paddlepaddle-gpu```



###### gpu-cuda9、10.0、10.1、11:

We only release paddlepaddle-gpu cuda10.2 on pypi.

If you want to install paddlepaddle-gpu with cuda version of 9.0 ,10.0 ,10.1 ,or 11.0, commands to install are on our website: [Installation Document](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/frompip_en.html)



#### Verify installation

After the installation is complete, you can use `python3` to enter the Python interpreter and then use `import paddle` and `paddle.utils.run_check()`

If `PaddlePaddle is installed successfully!` appears, to verify that the installation was successful.



### Other installation methods

If you want to install witch conda or docker or pip, please see commands to install on our website: [Installation Document](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)



## FOUR LEADING TECHNOLOGIES

- **Agile Framework for Industrial Development of Deep Neural Networks**

    The PaddlePaddle deep learning framework facilitates the development while lowering the technical burden, through leveraging a programmable scheme to architect the neural networks. It supports both declarative programming and imperative programming with both development flexibility and high runtime performance preserved.  The neural architectures could be automatically designed by algorithms with better performance than the ones designed by human experts.


-  **Support Ultra-Large-Scale Training of Deep Neural Networks**

    PaddlePaddle has made breakthroughs in ultra-large-scale deep neural networks training. It launched the world's first large-scale open-source training platform that supports the training of deep networks with 100 billions of features and trillions of parameters using data sources distributed over hundreds of nodes. PaddlePaddle overcomes the online deep learning challenges for ultra-large-scale deep learning models, and further achieved the real-time model updating with more than 1 trillion parameters.
     [Click here to learn more](https://github.com/PaddlePaddle/Fleet)


- **Accelerated High-Performance Inference over Ubiquitous Deployments**

    PaddlePaddle is not only compatible with other open-source frameworks for models training, but also works well on the ubiquitous developments, varying from platforms to devices. More specifically, PaddlePaddle accelerates the inference procedure with the fastest speed-up. Note that, a recent breakthrough of inference speed has been made by PaddlePaddle on Huawei's Kirin NPU, through the hardware/software co-optimization.
     [Click here to learn more](https://github.com/PaddlePaddle/Paddle-Lite)

- **Industry-Oriented Models and Libraries with Open Source Repositories**

     PaddlePaddle includes and maintains more than 100 mainstream models that have been practiced and polished for a long time in the industry. Some of these models have won major prizes from key international competitions. In the meanwhile, PaddlePaddle has further more than 200 pre-training models (some of them with source codes) to facilitate the rapid development of industrial applications.
     [Click here to learn more](https://github.com/PaddlePaddle/models)


## Documentation

We provide [English](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html) and
[Chinese](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html) documentation.

- [Basic Deep Learning Models](https://www.paddlepaddle.org.cn/documentation/docs/en/beginners_guide/index_en.html)

  You might want to start from how to implement deep learning basics with PaddlePaddle.


- [User Guides](https://www.paddlepaddle.org.cn/documentation/docs/en/user_guides/index_en.html)

  You might have got the hang of Beginner’s Guide, and wish to model practical problems and build your original networks.

- [Advanced User Guides](https://www.paddlepaddle.org.cn/documentation/docs/en/advanced_guide/index_en.html)

  So far you have already been familiar with Fluid. And the next step should be building a more efficient model or inventing your original Operator.


- [API Reference](https://www.paddlepaddle.org.cn/documentation/docs/en/api/index_en.html)

   Our new API enables much shorter programs.


- [How to Contribute](https://www.paddlepaddle.org.cn/documentation/docs/en/dev_guides/index_en.html)

   We appreciate your contributions!

## Communication

- [Github Issues](https://github.com/PaddlePaddle/Paddle/issues): bug reports, feature requests, install issues, usage issues, etc.
- QQ discussion group: 796771754 (PaddlePaddle).
- [Forums](https://aistudio.baidu.com/paddle/forum/): discuss implementations, research, etc.

## Copyright and License
PaddlePaddle is provided under the [Apache-2.0 license](LICENSE).
