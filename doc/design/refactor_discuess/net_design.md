# Overfiew

整体设计上参考 caffe2的SimpleNet，即Net 是一组 operator 的集合，包含了operator相关的操作，比如Create, RunOps, Delete 等。

Net 有如下一些特性：

- Net 管理其拥有的operator
- Net 本身不占用任何Variable资源
- Net 的调用方式类似 Functor，即 `output = net(inputs, scope)` ，其中
  - scope提供net执行所需要的全局Variable
  - 输入一个或多个variable, `inputs`
  - 返回一个variable, `output`

# API
