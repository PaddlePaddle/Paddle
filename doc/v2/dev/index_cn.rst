开发标准
========
PaddlePaddle遵守如下三个部分的代码和文档规范。

PaddlePaddle使用git做版本管理，docker作为构建和测试环境。代码中包含了Cuda, C++, Python, Shell等多种编程语言。语言规范遵守Google C++ Style, Pep-8, 代码库中包含自动化检查工具做风格检查。代码注释需要遵守Doxygen规范，不满足风格要求的代码会编译失败。关于如何使用git, 构建测试及代码开发, 我们提供了如下指南。

..  toctree::
  :maxdepth: 1

  contribute_to_paddle_cn.md

PaddlePaddle面向国内外用户，包含了中文和英文两部分的文档。设计文档和issue问题描述都推荐使用英文。对于设计文档，重在问题描述，背景阐述，然后才是解决方案。文档由Sphinx生成，因此代码注释也需要符合Sphinx文档标准。推荐本地使用paddlepaddle.org工具编译生成和预览文档，请参阅如下文档。

..  toctree::
  :maxdepth: 1

  write_docs_cn.rst

PaddlePaddle V2 使用新增Layer方式定义新的操作。组合基础API可以实现多种复杂Layer, 满足绝大多数应用。如需要定制Layer，请参阅如下文档，欢迎提交patch。

..  toctree::
  :maxdepth: 1

  new_layer_cn.rst
