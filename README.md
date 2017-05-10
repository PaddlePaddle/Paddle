# PaddlePaddle


[![Build Status](https://travis-ci.org/PaddlePaddle/Paddle.svg?branch=develop)](https://travis-ci.org/PaddlePaddle/Paddle)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](http://www.paddlepaddle.org/develop/doc/)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](http://www.paddlepaddle.org/doc_cn/)
[![Coverage Status](https://coveralls.io/repos/github/PaddlePaddle/Paddle/badge.svg?branch=develop)](https://coveralls.io/github/PaddlePaddle/Paddle?branch=develop)
[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle.svg)](https://github.com/PaddlePaddle/Paddle/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)


Welcome to the PaddlePaddle GitHub.

PaddlePaddle (PArallel Distributed Deep LEarning) is an easy-to-use,
efficient, flexible and scalable deep learning platform, which is originally
developed by Baidu scientists and engineers for the purpose of applying deep
learning to many products at Baidu.

Our vision is to enable deep learning for everyone via PaddlePaddle.
Please refer to our [release announcement](https://github.com/PaddlePaddle/Paddle/releases) to track the latest feature of PaddlePaddle.

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

It is recommended to check out the
[Docker installation guide](http://www.paddlepaddle.org/develop/doc/getstarted/build_and_install/docker_install_en.html)
before looking into the
[build from source guide](http://www.paddlepaddle.org/develop/doc/getstarted/build_and_install/build_from_source_en.html)

## Documentation

We provide [English](http://www.paddlepaddle.org/develop/doc/) and
[Chinese](http://www.paddlepaddle.org/doc_cn/) documentation.

- [Deep Learning 101](http://book.paddlepaddle.org/index.en.html)

  You might want to start from the this online interactive book that can run in Jupyter Notebook.

- [Distributed Training](http://www.paddlepaddle.org/develop/doc/howto/usage/cluster/cluster_train_en.html)

  You can run distributed training jobs on MPI clusters.

- [Distributed Training on Kubernetes](http://www.paddlepaddle.org/develop/doc/howto/usage/k8s/k8s_en.html)

   You can also run distributed training jobs on Kubernetes clusters.

- [Python API](http://www.paddlepaddle.org/develop/doc/api/index_en.html)

   Our new API enables much shorter programs.

- [How to Contribute](http://www.paddlepaddle.org/develop/doc/howto/dev/contribute_to_paddle_en.html)

   We appreciate your contributions!

## Ask Questions

You are welcome to submit questions and bug reports as [Github Issues](https://github.com/PaddlePaddle/Paddle/issues).

## Copyright and License
PaddlePaddle is provided under the [Apache-2.0 license](LICENSE).
