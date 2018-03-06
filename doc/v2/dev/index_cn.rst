开发标准
========
PaddlePaddle遵守如下三个部分的开发标准。
- 代码风格
  Paddle中包含了Cuda, C++, Python, Shell等多种编程语言。Cuda, C++的开发标准遵守Google C++ Style, 并加入了一些定制规则，https://github.com/PaddlePaddle/cpp-primer-digest。Python的开发标准遵守PEP-8标准, Shell遵守Google Shell Style。以上风格在提交代码时候, 代码库会通过pre-commit, clang-format自动化工具做风格检查。不满足风格要求的代码会编译失败。pre-commit也会自动format, 协助修改代码格式。

- 文档格式
  Paddle面向国内外用户，包含了中文和英文两部分的文档。设计文档和issue问题描述都推荐使用英文。对于设计文档，重在问题描述，背景阐述，然后才是解决方案。API文档由Sphinx生成，因此代码注释需要符合Sphinx文档标准。同样的，Paddle的集成测试工具会检测文档格式。推荐本地使用docker编译生成文档，本地修复文档。

- 框架定制
  Paddle V2使用新增Layer方式定义新的操作。定制Layer前请参阅已有的Layer, 如有通用性, 欢迎提交Layer实现。如何定制一个新的Layer见如下表。

此外，Paddle项目推荐使用docker作为开发环境。对于GPU环境，使用nvidia-docker，统一开发和集成环境。

..  toctree::
  :maxdepth: 1

  contribute_to_paddle_cn.md
  write_docs_cn.rst
  new_layer_cn.rst
