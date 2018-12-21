---
name: 预测（Inference Issue）
about: 您可以提问预测中报错、应用等问题。 You could use this template for reporting an inference issue.

---

为使您的问题得到快速解决，在建立Issue前，请您先通过如下方式搜索是否有相似问题:【搜索issue关键字】【使用labels筛选】【官方文档】

如果您没有查询到相似问题，为快速解决您的提问，建立issue时请提供如下细节信息：
- 标题：简洁、精准描述您的问题，例如“最新预测库的API文档在哪儿 ”
- 版本、环境信息：
    1）PaddlePaddle版本：请提供您的PaddlePaddle版本号（如1.1）或CommitID
    2）CPU：预测若用CPU，请提供CPU型号，MKL/OpenBlas/MKLDNN/等数学库使用情况
    3）GPU：预测若用GPU，请提供GPU型号、CUDA和CUDNN版本号
    4）系统环境：请您描述系统类型、版本（如Mac OS 10.14），Python版本
-预测信息
    1）C++预测：请您提供预测库安装包的版本信息，及其中的version.txt文件
    2）CMake包含路径的完整命令
    3）API信息（如调用请提供）
    4）预测库来源：官网下载/特殊环境（如BCLOUD编译）
- 复现信息：如为报错，请给出复现环境、复现步骤
- 问题描述：请详细描述您的问题，同步贴出报错信息、日志/代码关键片段

Thank you for contributing to PaddlePaddle.
Before submitting the issue, you could search issue in the github in case that th
If there is no solution,please make sure that this is an inference issue including the following details :
**System information**
-PaddlePaddle version （eg.1.1）or CommitID
-CPU: including CPUMKL/OpenBlas/MKLDNN version
-GPU: including CUDA/CUDNN version
-OS Platform (eg.Mac OS 10.14)
-Python version
-Cmake orders
-C++version.txt
-API information
**To Reproduce**
Steps to reproduce the behavior
**Describe your current behavior**
**Code to reproduce the issue**
**Other info / logs**
