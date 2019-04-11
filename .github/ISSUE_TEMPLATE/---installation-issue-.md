---
name: 安装（Installation Issue）
about: 您可以提问安装、编译出现报错等问题。 You could use this template for reporting an installation
   issue.

---

为使您的问题得到快速解决，在建立Issue前，请您先通过如下方式搜索是否有相似问题:【搜索issue关键字】【使用labels筛选】【官方文档】

建立issue时，为快速解决问题，请您根据使用情况给出如下信息：
- 标题：请包含关键词“安装错误”/“编译错误”，例如“Mac编译错误”
- 版本、环境信息：
    1）PaddlePaddle版本：请提供您的PaddlePaddle版本号（如1.1）或CommitID
    2）CPU：请提供CPU型号，MKL/OpenBlas/MKLDNN/等数学库的使用情况
    3）GPU：请提供GPU型号，CUDA和CUDNN版本号
    4）系统环境：请说明系统类型、版本（如Mac OS 10.14）、Python版本
- 安装方式信息：
1）pip安装/docker安装
2）本地编译：请提供cmake命令，编译命令
3）docker编译：请提供docker镜像，编译命令            
  特殊环境请注明：如离线安装等
- 复现信息：如为报错，请给出复现环境、复现步骤
- 问题描述：请详细描述您的问题，同步贴出报错信息、日志/代码关键片段

Thank you for contributing to PaddlePaddle.
Before submitting the issue, you could search issue in Github in case that there was a similar issue submitted or resolved before.
If there is no solution,please make sure that this is an installation issue including the following details:
**System information**
-PaddlePaddle version （eg.1.1）or CommitID
-CPU: including CPUMKL/OpenBlas/MKLDNN version
-GPU: including CUDA/CUDNN version
-OS Platform (eg. Mac OS 10.14)
-Python version
- Install method: pip install/install with docker/build from source(without docker)/build within docker
- Other special cases that you think may be related to this problem, eg. offline install, special internet condition   
**To Reproduce**
Steps to reproduce the behavior
**Describe your current behavior**
**Code to reproduce the issue**
**Other info / logs**
