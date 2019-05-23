# PaddlePaddle

[English](./README.md) | 简体中文

[![Build Status](https://travis-ci.org/PaddlePaddle/Paddle.svg?branch=develop)](https://travis-ci.org/PaddlePaddle/Paddle)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](http://www.paddlepaddle.org/documentation/docs/en/1.4/beginners_guide/index_en.html)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](http://www.paddlepaddle.org/documentation/docs/zh/1.4/beginners_guide/index_cn.html)
[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle.svg)](https://github.com/PaddlePaddle/Paddle/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

欢迎来到 PaddlePaddle GitHub

PaddlePaddle (PArallel Distributed Deep LEarning) 是一个简单易用、高效灵活、可扩展的深度学习平台，最初由百度科学家和工程师共同开发，目的是将深度学习技术应用到百度的众多产品中。

我们的愿景是让每个人都能通过PaddlePaddle接触深度学习

跟进PaddlePaddle最新特性请参考我们的[版本说明](https://github.com/PaddlePaddle/Paddle/releases)

### PaddlePaddle最新版本: [Fluid 1.4.1](https://github.com/PaddlePaddle/Paddle/tree/release/1.4)
### 安装最新稳定版本:
```
# Linux CPU
pip install paddlepaddle
# Linux GPU cuda9cudnn7
pip install paddlepaddle-gpu
# Linux GPU cuda8cudnn7
pip install paddlepaddle-gpu==1.4.1.post87
# Linux GPU cuda8cudnn5
pip install paddlepaddle-gpu==1.4.1.post85

# 其他平台上的安装指引请参考 http://paddlepaddle.org/
```

## 特性

- **灵活性**

    PaddlePaddle支持丰富的神经网络架构和优化算法。易于配置复杂模型，例如带有注意力机制或复杂记忆连接的神经网络机器翻译模型。

-  **高效性**

    为了高效使用异步计算资源，PaddlePaddle对框架的不同层进行优化，包括计算、存储、架构和通信。下面是一些样例：

    - 通过SSE/AVX 内置函数、BLAS库(例如MKL、OpenBLAS、cuBLAS)或定制的CPU/GPU内核优化数学操作。
    - 通过MKL-DNN库优化CNN网络
    - 高度优化循环网络，无需执行 `padding` 操作即可处理 **变长** 序列
    - 针对高维稀疏数据模型，优化了局部和分布式训练。


- **稳定性**

    有了 PaddlePaddle，使得利用各种CPU/GPU和机器来加速训练变得简单。PaddlePaddle 通过优化通信可以实现巨大吞吐量和快速执行。

- **与产品相连**

    另外，PaddlePaddle 的设计也易于部署。在百度，PaddlePaddle 已经部署到含有巨大用户量的产品和服务上，包括广告点击率（CTR）预测、大规模图像分类、光学字符识别（OCR）、搜索排序，计算机病毒检测、推荐系统等等。PaddlePaddle广泛应用于百度产品中，产生了非常重要的影响。我们希望您也能探索 PaddlePaddle 的能力，为您的产品创造新的影响力和效果。

## 安装

推荐阅读官网上的[安装说明](http://www.paddlepaddle.org/documentation/docs/zh/1.4/beginners_guide/install/index_cn.html)

## 文档

我们提供[英文](http://www.paddlepaddle.org/documentation/docs/en/1.4/beginners_guide/index_en.html)和
[中文](http://www.paddlepaddle.org/documentation/docs/zh/1.4/beginners_guide/install/index_cn.html) 文档

- [深度学习101](https://github.com/PaddlePaddle/book)

  或许您想从这个在线交互式书籍开始，可以在Jupyter Notebook中运行

- [分布式训练](http://paddlepaddle.org/documentation/docs/zh/1.4/user_guides/howto/training/multi_node.html)

  可以在MPI集群上运行分布式训练任务

- [Python API](http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/index_cn.html)

   新的API支持代码更少更简洁的程序

- [贡献方式](http://paddlepaddle.org/documentation/docs/zh/1.4/advanced_usage/development/contribute_to_paddle/index_cn.html)

   欢迎您的贡献!

## 答疑

欢迎您将问题和bug报告以[Github Issues](https://github.com/PaddlePaddle/Paddle/issues)的形式提交

## 版权和许可证
PaddlePaddle由[Apache-2.0 license](LICENSE)提供
