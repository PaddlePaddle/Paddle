开发标准
========
PaddlePaddle遵守如下三个部分的代码和文档规范

- 代码风格
  PaddlePaddle中包含了Cuda, C++, Python, Shell等多种编程语言。语言规范遵守Google C++ Style, Pep-8, 代码库中包含自动化检查工具做风格检查。不满足风格要求的代码会编译失败。pre-commit也会自动format, 协助修改代码格式。

..  toctree::
  :maxdepth: 1

  contribute_to_paddle_cn.md

- 文档格式
  PaddlePaddle面向国内外用户，包含了中文和英文两部分的文档。设计文档和issue问题描述都推荐使用英文。对于设计文档，重在问题描述，背景阐述，然后才是解决方案。API文档由Sphinx生成，因此代码注释需要符合Sphinx文档标准。同样的，PaddlePaddle的集成测试工具会检测文档格式。推荐本地使用docker编译生成文档，本地修复文档。

..  toctree::
  :maxdepth: 1

  write_docs_cn.rst

- 框架定制
  PaddlePaddle V2使用新增Layer方式定义新的操作。组合基础api可以实现多种复杂Layer, 满足绝大多数应用。如需要定制Layer，请参阅如下文档，欢迎提交patch.

..  toctree::
  :maxdepth: 1

  new_layer_cn.rst

此外，PaddlePaddle推荐使用docker作为开发环境。对于GPU环境，使用nvidia-docker，统一开发和集成环境。
