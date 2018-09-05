#############
如何贡献文档
#############

PaddlePaddle非常欢迎您贡献文档。如果您撰写/翻译的文档满足我们的要求，您的文档将会呈现在paddlapaddle.org网站和Github上供PaddlePaddle的用户阅读。

Paddle的文档主要分为以下几个模块：

- 新手入门：包括安装说明、深度学习基础知识、学习资料等，旨在帮助用户快速安装和入门；

- 使用指南：包括数据准备、网络配置、训练、Debug、预测部署和模型库文档，旨在为用户提供PaddlePaddle基本用法讲解；

- 进阶使用：包括服务器端和移动端部署、如何贡献代码/文档、如何性能调优等，旨在满足开发者的需求；

我们的文档支持 `reStructured Text <http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ 和 `Markdown <https://guides.github.com/features/mastering-markdown/>`_ (GitHub风格)格式的内容贡献。

撰写文档完成后，您可以使用预览工具查看文档在官网显示的效果，以验证您的文档是否能够在官网正确显示。


如何使用预览工具
=================

1. Clone你希望更新或测试的相关仓库：
-------------------------------------

*如果您已经拥有了这些存储库的本地副本，请跳过此步骤*

可拉取的存储库有：

..  code-block:: bash

    git clone https://github.com/PaddlePaddle/Paddle.git
    git clone https://github.com/PaddlePaddle/book.git
    git clone https://github.com/PaddlePaddle/models.git
    git clone https://github.com/PaddlePaddle/Mobile.git

您可以将这些本地副本放在电脑的任意目录下，稍后我们会在启动 PaddlePaddle.org时指定这些仓库的位置。

2. 在新目录下拉取 PaddlePaddle.org 并安装其依赖项
--------------------------------------------------

在此之前，请确认您的操作系统安装了python的依赖项

以ubuntu系统为例，运行：

..  code-block:: bash

    sudo apt-get update && apt-get install -y python-dev build-essential


然后：

..  code-block:: bash

    git clone https://github.com/PaddlePaddle/PaddlePaddle.org.git
    cd PaddlePaddle.org/portal
    # To install in a virtual environment.
    # virtualenv venv; source venv/bin/activate
    pip install -r requirements.txt


**可选项**：如果你希望实现中英网站转换，以改善PaddlePaddle.org，请安装 `GNU gettext <https://www.gnu.org/software/gettext/>`_

3. 在本地运行 PaddlePaddle.org
--------------------------------------------------

添加您希望加载和构建内容的目录列表(选项包括： ``--paddle``， ``--book``， ``--models``， ``--mobile``)

运行：

..  code-block:: bash

    ./runserver --paddle <path_to_paddle_dir> --book <path_to_book_dir>


**注意：**  `<pathe_to_paddle_dir>` 为第一步中paddle副本在您本机的存储地址，并且对于 --paddle目录，您可以指向特定的API版本目录（例如： `<path to Paddle>/doc/fluid` or `<path to Paddle>v2` )

然后：

打开浏览器并导航到 `http://localhost:8000 <http://localhost:8000>`_ 。

*网站可能需要几秒钟才能成功加载，因为构建需要一定的时间。*

如何书写文档
============

PaddlePaddle文档使用 `sphinx <http://www.sphinx-doc.org/en/1.4.8/>`_ 自动生成，用户可以参考sphinx教程进行书写。

贡献新的内容
============

所有存储库都支持 `Markdown <https://guides.github.com/features/mastering-markdown/>`_ (GitHub风格)格式的内容贡献，同时也支持 `reStructured Text <http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ 格式。

在完成安装步骤后，您还需要完成下列操作：

- 在你开始写作之前，我们建议你回顾一下这些关于贡献内容的指南。
- 创建一个新的 `.md` 文件（在 paddle repo 中可以创建 `.rst` 文件）或者在您当前操作的仓库中修改已存在的文章。
- 查看浏览器中的更改，请单击右上角的Refresh Content。
- 将修改的文档添加到菜单或更改其在菜单上的位置，请单击页面左侧菜单顶部的Edit menu按钮，打开菜单编辑器。

贡献或修改Python API
=======================

在build了新的pybind目标并测试了新的Python API之后，您可以继续测试文档字符串和注释的显示方式:

- 我们建议回顾这些API文档贡献指南。
- 确保构建的Python目录(包含 Paddle )在您运行`./runserver`的Python路径中可用。
- 在要更新的特定“API”页面上，单击右上角的Refresh Content。
- 将修改的API添加到菜单或更改其在菜单上的位置，请单击页面左侧菜单顶部的Edit menu按钮，打开菜单编辑器。

帮助改进预览工具
=======================

我们非常欢迎您对平台和支持内容的各个方面做出贡献，以便更好地呈现这些内容。您可以Fork或Clone这个存储库，或者提出问题并提供反馈，以及在issues上提交bug信息。详细内容请参考 `开发指南 <https://github.com/PaddlePaddle/PaddlePaddle.org/blob/develop/DEVELOPING.md>`_ 。

版权和许可
=======================
PaddlePaddle.org在Apache-2.0的许可下提供