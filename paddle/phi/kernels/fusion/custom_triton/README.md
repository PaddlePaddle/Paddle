OpenAI Triton为专用领域的编程语言与编译器，巧妙地在硬件算子性能和开发效率之间达到了均衡。
Paddle提供了简单的示例，展示了如何将用户编写的Trtion自定义算子，接入Paddle。

# 如何利用Triton生成一个 cuda kernel

用户只需要在py文件中编写一个被`@triton.jit`装饰的核函数，即可实现一个Triton算子。
借助Triton的AOT工具，可以编译出cubin kernel。
下面以`matmul_triton.py`为例，展示如何利用Triton生成一个cubin kernel。

1. 用户在`matmul_triton.py`编写一个由`@triton.jit`装饰的核函数。

2. 利用如下命令产出cubin kernel。













