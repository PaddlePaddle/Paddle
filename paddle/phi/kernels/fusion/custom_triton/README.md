

OpenAI Triton为专用领域的编程语言与编译器，巧妙地在硬件算子性能和开发效率之间达到了均衡。
Paddle提供了简单的示例，展示了如何将用户编写的Trtion自定义算子，接入Paddle。

# 如何添加一个Triton算子


## 编写Triton算子

用Python文件写好一个triton算子，借助于triton的aot工具，可以编译出cubin kernel。





