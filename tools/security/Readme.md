# PaddlePaddle代码漏洞挖掘工具

PaddlePaddle深度学习框架代码漏洞挖掘工具示例。

项目主要包含两部分：**Fuzzing**和**代码审计**

## Fuzzing

Fuzzing部分为使用atheris+libfuzzer对Paddle op kernel进行模糊测试的工具。

## 代码审计

代码审计部分包括CodeQL研究相关示例代码，可用于静态扫描找到特定类型的漏洞。
