# PaddlePaddle

English | [简体中文](./README_cn.md)

[![Build Status](https://travis-ci.org/PaddlePaddle/Paddle.svg?branch=develop)](https://travis-ci.org/PaddlePaddle/Paddle)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](http://www.paddlepaddle.org.cn/documentation/docs/en/1.6/beginners_guide/index_en.html)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](http://www.paddlepaddle.org.cn/documentation/docs/zh/1.6/beginners_guide/index_cn.html)
[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle.svg)](https://github.com/PaddlePaddle/Paddle/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

Welcome to the PaddlePaddle GitHub.

PaddlePaddle (PArallel Distributed Deep LEarning) is an easy-to-use,
efficient, flexible and scalable deep learning platform, which is originally
developed by Baidu scientists and engineers for the purpose of applying deep
learning to many products at Baidu.

Our vision is to enable deep learning for everyone via PaddlePaddle.
Please refer to our [release announcement](https://github.com/PaddlePaddle/Paddle/releases) to track the latest feature of PaddlePaddle.

### Latest PaddlePaddle Release: [v1.6](https://github.com/PaddlePaddle/Paddle/tree/release/1.6)
### Install Latest Stable Release:
```
# Linux CPU
pip install paddlepaddle
# Linux GPU cuda10cudnn7
pip install paddlepaddle-gpu
# Linux GPU cuda9cudnn7
pip install paddlepaddle-gpu==1.6.0.post97


# For installation on other platform, refer to http://paddlepaddle.org/
```
Now our developers could acquire Tesla V100 online computing resources for free. If you create a program by AI Studio, you would obtain 12 hours to train models online per day. If you could insist on that for five consecutive days, then you would own extra 48 hours. [Click here to start](http://ai.baidu.com/support/news?action=detail&id=981).

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
      (e.g. MKL, OpenBLAS, cuBLAS) or customized CPU/GPU kernels.
      - Optimized CNN networks through MKL-DNN library.
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
    PaddlePaddle has been deployed into products and services with a vast number
    of users, including ad click-through rate (CTR) prediction, large-scale image
    classification, optical character recognition(OCR), search ranking, computer
    virus detection, recommendation, etc. It is widely utilized in products at
    Baidu and it has achieved a significant impact. We hope you can also explore
    the capability of PaddlePaddle to make an impact on your product.

## Installation

It is recommended to read [this doc](http://www.paddlepaddle.org.cn/documentation/docs/en/1.6/beginners_guide/index_en.html) on our website.

## Documentation

We provide [English](http://www.paddlepaddle.org.cn/documentation/docs/en/1.6/beginners_guide/index_en.html) and
[Chinese](http://www.paddlepaddle.org.cn/documentation/docs/zh/1.6/beginners_guide/install/index_cn.html) documentation.

- [Deep Learning 101](https://github.com/PaddlePaddle/book)

  You might want to start from this online interactive book that can run in a Jupyter Notebook.

- [Distributed Training](http://paddlepaddle.org.cn/documentation/docs/en/1.6/user_guides/howto/training/multi_node_en.html)

  You can run distributed training jobs on MPI clusters.

- [Python API](http://paddlepaddle.org.cn/documentation/docs/en/1.6/api/index_en.html)

   Our new API enables much shorter programs.

- [How to Contribute](http://paddlepaddle.org.cn/documentation/docs/en/1.6/advanced_usage/development/contribute_to_paddle/index_en.html)

   We appreciate your contributions!

## Communication

- [Github Issues](https://github.com/PaddlePaddle/Paddle/issues): bug reports, feature requests, install issues, usage issues, etc.
- QQ discussion group: 796771754 (PaddlePaddle).
- [Forums](http://ai.baidu.com/forum/topic/list/168?pageNo=1): discuss implementations, research, etc.

## Copyright and License
PaddlePaddle is provided under the [Apache-2.0 license](LICENSE).
