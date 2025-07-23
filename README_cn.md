
<p align="center">
<img align="center" src="doc/imgs/logo.png", width=1600>
<p>

--------------------------------------------------------------------------------

[English](./README.md) | 简体中文 | [日本語](./README_ja.md)

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)
[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle.svg)](https://github.com/PaddlePaddle/Paddle/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

欢迎来到 PaddlePaddle GitHub

飞桨(PaddlePaddle)以百度多年的深度学习技术研究和业务应用为基础，是中国首个自主研发、功能完备、 开源开放的产业级深度学习平台，集深度学习核心训练和推理框架、基础模型库、端到端开发套件和丰富的工具组件于一体。目前，飞桨文心开发者数量已超过2185万，服务超过67万家企业，创建的模型达到110万。飞桨助力开发者快速实现 AI 想法，快速上线 AI 业务。帮助越来越多的行业完成 AI 赋能，实现产业智能化升级。

## 安装

### PaddlePaddle 最新版本: [3.1](https://github.com/PaddlePaddle/Paddle/tree/release/3.1)

跟进 PaddlePaddle 最新特性请参考我们的[版本说明](https://github.com/PaddlePaddle/Paddle/releases)

### 安装最新稳定版本

``` sh
# CPU
pip install paddlepaddle
# GPU
pip install paddlepaddle-gpu
```

更多安装信息详见官网 [安装说明](https://www.paddlepaddle.org.cn/install/quick)。

## 飞桨新一代框架 3.1

- **动静统一自动并行**

  只需在单卡基础上进行少量的张量切分标记，飞桨能自动寻找最⾼效的分布式并行策略，大幅度降低了产业开发和训练的成本，使开发者能够更专注于模型和算法的创新。

- **大模型训练推一体**

  同一套框架支持训练和推理，实现训练、推理代码复用和无缝衔接，为大模型的全流程提供了统一的开发体验和极致的训练效率，为产业提供了极致的开发体验。

- **科学计算高阶微分**

  提供高阶自动微分、复数运算、傅里叶变换、编译优化、分布式训练等能力支持，支持数学、力学、材料、气象、生物等领域科学探索，微分方程求解速度大幅提升。

- **神经网络编译器**

  采用框架一体化设计，支持⽣成式模型、科学计算模型等多种模型的高效训练与可变形推理，在计算灵活性与高性能之间提供了良好的平衡点，显著降低性能优化成本。

- **异构多芯适配**

  成熟且完整的多硬件统一适配方案，通过标准化接口屏蔽了不同芯片软件栈开发接口差异，实现可插拔架构。

## 文档

我们提供 [英文](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html) 和 [中文](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html) 文档

- [使用指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)：或许你想从深度学习基础开始学习飞桨

- [应用实践](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/index_cn.html)：使用飞桨搭建你的模型，更高效的完成深度学习任务

- [API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html)：新的 API 支持代码更少更简洁的程序

- [贡献方式](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/08_contribution/index_cn.html)：参与飞桨社区开源贡献的指南

## 开源社区

- [Github Issues](https://github.com/PaddlePaddle/Paddle/issues)：错误报告、功能请求、安装问题、使用问题等。
- 我们的许多贡献活动都提供来自经验丰富的社区成员的不同程度的指导，请查看置顶的 issues 中的活动，并考虑参加。
- 社区博客：[https://pfcc.blog/](https://pfcc.blog/)。
- 了解更多详情：[Community](https://github.com/PaddlePaddle/community)。

## 版权和许可证

PaddlePaddle 由[Apache-2.0 license](LICENSE)提供
