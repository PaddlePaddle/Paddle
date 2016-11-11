# PaddlePaddle


[![Build Status](https://travis-ci.org/baidu/Paddle.svg?branch=master)](https://travis-ci.org/baidu/Paddle)
[![Coverage Status](https://coveralls.io/repos/github/baidu/Paddle/badge.svg?branch=develop)](https://coveralls.io/github/baidu/Paddle?branch=develop)
[![Join the chat at https://gitter.im/PaddlePaddle/Deep_Learning](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/PaddlePaddle/Deep_Learning?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Welcome to the PaddlePaddle GitHub.

PaddlePaddle (PArallel Distributed Deep LEarning) is an easy-to-use,
efficient, flexible and scalable deep learning platform, which is originally
developed by Baidu scientists and engineers for the purpose of applying deep
learning to many products at Baidu.

Our vision is to enable deep learning for everyone via PaddlePaddle.
Please refer to our [release announcement](https://github.com/baidu/Paddle/releases) to track the latest feature of PaddlePaddle. 

## Features

- **Flexibility**

    PaddlePaddle supports a wide range of neural network architectures and
    optimization algorithms. It is easy to configure complex models such as
    neural machine translation model with attention mechanism or complex memory
    connection.

-  **Efficiency**
  
    In order to unleash the power of heterogeneous computing resource,
    optimization occurs at different levels of PaddlePaddle, including
    computing, memory, architecture and communication. The following are some
    examples:

      - Optimized math operations through SSE/AVX intrinsics, BLAS libraries
      (e.g. MKL, ATLAS, cuBLAS) or customized CPU/GPU kernels. 
      - Highly optimized recurrent networks which can handle **variable-length** 
      sequence without padding.
      - Optimized local and distributed training for models with high dimensional
      sparse data.

- **Scalability**

    With PaddlePaddle, it is easy to use many CPUs/GPUs and machines to speed
    up your training. PaddlePaddle can achieve high throughput and performance
    via optimized communication.

- **Connected to Products**

    In addition, PaddlePaddle is also designed to be easily deployable. At Baidu,
    PaddlePaddle has been deployed into products or service with a vast number
    of users, including ad click-through rate (CTR) prediction, large-scale image
    classification, optical character recognition(OCR), search ranking, computer
    virus detection, recommendation, etc. It is widely utilized in products at
    Baidu and it has achieved a significant impact. We hope you can also exploit
    the capability of PaddlePaddle to make a huge impact for your product.

## Installation
Check out the [Install Guide](http://paddlepaddle.org/doc/build/) to install from
pre-built packages (**docker image**, **deb package**) or 
directly build on **Linux** and **Mac OS X** from the source code.
 
## Documentation
Both [English Docs](http://paddlepaddle.org/doc/) and [Chinese Docs](http://paddlepaddle.org/doc_cn/) are provided for our users and developers.

- [Quick Start](http://paddlepaddle.org/doc/demo/quick_start/index_en) <br>
   You can follow the quick start tutorial to learn how use PaddlePaddle
   step-by-step.
    
- [Example and Demo](http://paddlepaddle.org/doc/demo/) <br>
   We provide five demos, including: image classification, sentiment analysis,
   sequence to sequence model, recommendation, semantic role labeling. 
   
- [Distributed Training](http://paddlepaddle.org/doc/cluster) <br>
  This system supports training deep learning models on multiple machines
  with data parallelism.
   
- [Python API](http://paddlepaddle.org/doc/ui/) <br>
   PaddlePaddle supports using either Python interface or C++ to build your
   system. We also use SWIG to wrap C++ source code to create a user friendly
   interface for Python. You can also use SWIG to create interface for your
   favorite programming language.
 
- [How to Contribute](http://paddlepaddle.org/doc/build/contribute_to_paddle.html) <br>
   We sincerely appreciate your interest and contributions. If you would like to
   contribute, please read the contribution guide.   

- [Source Code Documents](http://paddlepaddle.org/doc/source/) <br>

## Ask Questions
Please join the [**gitter chat**](https://gitter.im/PaddlePaddle/Deep_Learning) or send email to
**paddle-dev@baidu.com** to ask questions and talk about methods and models.
Framework development discussions and
bug reports are collected on [Issues](https://github.com/baidu/paddle/issues).

## Copyright and License
PaddlePaddle is provided under the [Apache-2.0 license](LICENSE).
