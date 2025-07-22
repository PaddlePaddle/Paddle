<p align="center">
<img align="center" src="doc/imgs/logo.png", width=1600>
<p>

--------------------------------------------------------------------------------

English | [简体中文](./README_cn.md) | [日本語](./README_ja.md)

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)
[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle.svg)](https://github.com/PaddlePaddle/Paddle/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FPaddlePaddle)

Welcome to the PaddlePaddle GitHub.

PaddlePaddle, as the first independent R&D deep learning platform in China, has been officially open-sourced to professional communities since 2016. It is an industrial platform with advanced technologies and rich features that cover core deep learning frameworks, basic model libraries, end-to-end development kits, tools & components as well as service platforms.
PaddlePaddle originates from industrial practices with dedication and commitments to industrialization. It has been widely adopted by a wide range of sectors including manufacturing, agriculture, enterprise service, and so on while serving more than 21.85 million developers, 670,000 companies and generating 1,100,000 models. With such advantages, PaddlePaddle has helped an increasing number of partners commercialize AI.

## Installation

### Latest PaddlePaddle Release: [3.1](https://github.com/PaddlePaddle/Paddle/tree/release/3.1)

Our vision is to enable deep learning for everyone via PaddlePaddle.
Please refer to our [release announcement](https://github.com/PaddlePaddle/Paddle/releases) to track the latest features of PaddlePaddle.

### Install Latest Stable Release

``` sh
# CPU
pip install paddlepaddle
# GPU
pip install paddlepaddle-gpu
```

For more information about installation, please view [Quick Install](https://www.paddlepaddle.org.cn/install/quick)

## **PaddlePaddle New Generation Framework 3.1**

* **Unified Dynamic/Static Graphs and Automatic Parallelism**

    By requiring only minimal tensor partitioning annotations based on a single-card configuration, PaddlePaddle automatically discovers the most efficient distributed parallel strategy. This significantly reduces the costs of industrial development and training, enabling developers to focus more intently on model and algorithm innovation.

* **Integrated Training and Inference for Large Models**

    The same framework supports both training and inference, achieving code reuse and seamless integration between these stages. This provides a unified development experience and maximum training efficiency for the entire large model workflow, offering the industry a superior development experience.

* **High-Order Differentiation for Scientific Computing**

    Provides capabilities such as high-order automatic differentiation, complex number operations, Fourier transforms, compilation optimization, and distributed training support. It facilitates scientific exploration in fields including mathematics, mechanics, materials science, meteorology, and biology, substantially improving the speed of solving differential equations.

* **Neural Network Compiler**

    Adopting an integrated framework design, it supports efficient training and flexible inference for diverse models, including generative and scientific computing models. It achieves an effective balance between computational flexibility and high performance, significantly lowering performance optimization costs.

* **Heterogeneous Multi-Chip Adaptation**
    Features a mature and complete unified adaptation solution for multiple hardware types. Through standardized interfaces, it abstracts the variations in development interfaces across different chip software stacks, realizing a pluggable architecture.

## Documentation

We provide [English](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html) and
[Chinese](https://www.paddlepaddle.org.cn/documentation/docs/zh/guide/index_cn.html) documentation.

* [Guides](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)

  You might want to start from how to implement deep learning basics with PaddlePaddle.

* [Practice](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/index_cn.html)

  So far you have already been familiar with Fluid. And the next step should be building a more efficient model or inventing your original Operator.

* [API Reference](https://www.paddlepaddle.org.cn/documentation/docs/en/api/index_en.html)

   Our new API enables much shorter programs.

* [How to Contribute](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/08_contribution/index_en.html)

   We appreciate your contributions!

## Open Source Community

* [Github Issues](https://github.com/PaddlePaddle/Paddle/issues): bug reports, feature requests, install issues, usage issues, etc.
* Many of our contribution events offer varying levels of mentorship from experienced community members, please check the events in the pinned issues, and consider attending.
* Community Blog: <https://pfcc.blog/>
* See more details about PaddlePaddle community at [community](https://github.com/PaddlePaddle/community).

## Copyright and License

PaddlePaddle is provided under the [Apache-2.0 license](LICENSE).
