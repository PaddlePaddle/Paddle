---
name: 训练（Training issue）
about: 您可以提问训练中报错、应用、出core等问题。 You could use this template for reporting an training
   issue.

---

为使您的问题得到快速解决，在建立Issues前，请您先通过如下方式搜索是否有相似问题:【搜索issue关键字】【使用labels筛选】【官方文档】

如果您没有查询到相似问题，为快速解决您的提问，建立issue时请提供如下细节信息：
- 标题：简洁、精准概括您的问题，例如“Insufficient Memory xxx" ”
- 版本、环境信息：
    1）PaddlePaddle版本：请提供您的PaddlePaddle版本号，例如1.1或CommitID
    2）CPU：预测若用CPU，请提供CPU型号，MKL/OpenBlas/MKLDNN/等数学库使用情况
    3）GPU：预测若用GPU，请提供GPU型号、CUDA和CUDNN版本号
    4）系统环境：请您描述系统类型、版本，例如Mac OS 10.14，Python版本
- 训练信息
    1）单机/多机，单卡/多卡
    2）显存信息
    3）Operator信息
- 复现信息：如为报错，请给出复现环境、复现步骤
- 问题描述：请详细描述您的问题，同步贴出报错信息、日志、可复现的代码片段

Thank you for contributing to PaddlePaddle.
Before submitting the issue, you could search issue in the github in case that there was a similar issue submitted or resolved before.
If there is no solution,please make sure that this is a training issue including the following details:
**System information**
-PaddlePaddle version （eg.1.1）or CommitID
-CPU: including CPUMKL/OpenBlas/MKLDNN version
-GPU: including CUDA/CUDNN version
-OS Platform (eg.Mac OS 10.14)
-Other imformation: Distriuted training/informantion of operator/
Graphics card storage
**To Reproduce**
Steps to reproduce the behavior
**Describe your current behavior**
**Code to reproduce the issue**
**Other info / logs**
