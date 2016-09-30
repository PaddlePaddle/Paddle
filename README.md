# PaddlePaddle
[![Build Status](https://travis-ci.org/baidu/Paddle.svg?branch=master)](https://travis-ci.org/baidu/Paddle)

Welcome to the PaddlePaddle GitHub.

The software will be released on Sept. 30 with full documentation and installation support. 

A pre-release version is available now for those who are eager to take a look.

PaddlePaddle (PArallel Distributed Deep LEarning) is an easy-to-use,
efficient, flexible and scalable deep learning platform, which is originally
developed by Baidu scientists and engineers for the purpose of applying deep
learning to many products at Baidu.

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
  1. Optimized math operations through SSE/AVX intrinsics, BLAS libraries
  (e.g. MKL, ATLAS, cuBLAS) or customized CPU/GPU kernels. 
  2. Highly optimized recurrent networks which can handle **variable-length** 
  sequence without padding.
  3. Optimized local and distributed training for models with high dimensional
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
See [Installation Guide](http://paddlepaddle.org/doc/build/) to install from pre-built package or build from the source code. (Note: The installation packages are still in pre-release state and your experience of installation may not be smooth.).

## Documentation
- [Chinese Documentation](http://paddlepaddle.org/doc_cn/) <br>

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

If you want to ask questions and discuss about methods and models, welcome
to send email to paddle-dev@baidu.com. Framework development discussions and
bug reports are collected on [Issues](https://github.com/baidu/paddle/issues).

## Copyright and License
PaddlePaddle is provided under the [Apache-2.0 license](LICENSE).
