# 如何写新的Operator

 - [概念简介](#概念简介)
 - [实现C++类](#实现c类)
   - [定义ProtoMaker类](#定义protomaker类)
   - [定义Operator类](#定义operator类)
   - [定义OpKernel类](#定义opkernel类)
   - [注册Operator](#注册operator)
   - [编译](#编译)
 - [绑定Python](#绑定python)
 - [实现单元测试](#实现单元测试)
   - [前向Operator单测](#前向operator单测)
   - [反向Operator单测](#反向operator单测)
   - [编译和执行](#编译和执行)
 - [注意事项](#注意事项)


## 概念简介

简单介绍需要用到基类，详细介绍请参考设计文档。

- `framework::OperatorBase`: Operator(简写，Op)基类。
- `framework::OpKernel`: Op计算函数的基类，称作Kernel。
- `framework::OperatorWithKernel`：继承自OperatorBase，Op有计算函数，称作有Kernel。
- `class OpProtoAndCheckerMaker`：描述该Op的输入、输出、属性、注释,主要用于Python API接口生成

依据是否包含kernel，可以将Op分为两种：包含Kernel的Op和不包含kernel的Op，前者Op的定义继承自`OperatorWithKernel`，后者继承自`OperatorBase`。本教程主要介绍带Kernel的Op如何写，简单总结Op需要包含的内容如下：

<table>
<thead>
<tr>
<th>内容</th>
<th>定义位置</th>
</tr>
</thead>
<tbody>
<tr>
<td>OpProtoMake定义 </td>
<td>`.cc`文件，Backward Op不需要定义OpProtoMake </td>
</tr>
<tr>
<td>Op定义 </td>
<td> `.cc`文件</td>
</tr>
<tr>
<td>Kernel实现 </td>
<td> CPU、CUDA共享Kernel实现在`.h`文件中，否则，CPU 实现在`.cc`文件中，CUDA 实现在`.cu`文件中。</td>
</tr>
<tr>
<td>注册Op </td>
<td> Op注册实现在`.cc`文件；Kernel注册CPU实现在`.cc`文件中，CUDA实现在`.cu`文件中</td>
</tr>
</tbody>
</table>


实现新的op都添加至目录[paddle/fluid/operators](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/operators)下，文件命名以`*_op.h`（如有） 、 `*_op.cc` 、`*_op.cu`（如有）结尾。**系统会根据文件名自动构建op和其对应的Python扩展。**


下面以矩阵乘操作，即[MulOp](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/mul_op.cc)为例来介绍如何写带Kernel的Operator。


## 实现C++类


### 定义ProtoMaker类

矩阵乘法的公式：$Out = X * Y$, 可见该计算由两个输入，一个输出组成。

首先定义`ProtoMaker`来描述该Op的输入、输出，并添加注释：

```cpp
class MulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MulOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor), 2D tensor of size (M x K)");
    AddInput("Y", "(Tensor), 2D tensor of size (K x N)");
    AddOutput("Out", "(Tensor), 2D tensor of size (M x N)");
    AddComment(R"DOC(
Two Element Mul Operator.
The equation is: Out = X * Y
)DOC");
  }
};
```

[`MulOpMaker`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/mul_op.cc#L76-L127)继承自`framework::OpProtoAndCheckerMaker`，构造函数含有2个参数：

   - `framework::OpProto` ： 前者存储Op的输入输出和参数属性，将用于Python API接口的生成。
   - `framework::OpAttrChecker` ：后者用于检查参数属性的合法性。

构造函数里通过`AddInput`添加输入参数，通过`AddOutput`添加输出参数，通过`AddComment`添加Op的注释。这些函数会将对应内容添加到`OpProto`中。

上面的代码在`MulOp`中添加两个输入`X`和`Y`，添加了一个输出`Out`，并解释了各自含义，命名请遵守[命名规范](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/dev/name_convention.md)。


再以[`ScaleOp`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/scale_op.cc#L38-L55)为例：

```cpp
template <typename AttrType>
class ScaleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ScaleOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor) Input tensor of scale operator.");
    AddOutput("Out", "(Tensor) Output tensor of scale operator.");
    AddComment(R"DOC(
Scale operator
$$Out = scale*X$$
)DOC");
    AddAttr<AttrType>("scale",
                      "(float, default 1.0)"
                      "The scaling factor of the scale operator.")
        .SetDefault(1.0);
  }
};
```

这个例子有`AddAttr<AttrType>("scale", "...").SetDefault(1.0);` : 增加`scale`系数，作为参数属性，并且设置默认值为1.0。


### 定义Operator类

下面的点实现了MulOp的定义：

```cpp
class MulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto dim0 = ctx.Input<Tensor>("X")->dims();
    auto dim1 = ctx.Input<Tensor>("Y")->dims();
    PADDLE_ENFORCE_EQ(dim0.size(), 2,
                      "input X(%s) should be a tensor with 2 dims, a matrix",
                      ctx.op_.Input("X"));
    PADDLE_ENFORCE_EQ(dim1.size(), 2,
                      "input Y(%s) should be a tensor with 2 dims, a matrix",
                      ctx.op_.Input("Y"));
    PADDLE_ENFORCE_EQ(
        dim0[1], dim1[0],
        "First matrix's width must be equal with second matrix's height.");
    ctx.Output<Tensor>("Out")->Resize({dim0[0], dim1[1]});
  }
};
```

[`MulOp`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/mul_op.cc#L22)继承自`OperatorWithKernel`。`public`成员：

```cpp
using framework::OperatorWithKernel::OperatorWithKernel;
```

这句表示使用基类`OperatorWithKernel`的构造函数，也可写成：

```cpp
MulOp(const std::string &type, const framework::VariableNameMap &inputs,
      const framework::VariableNameMap &outputs,
      const framework::AttributeMap &attrs)
  : OperatorWithKernel(type, inputs, outputs, attrs) {}
```

还需要重写`InferShape`接口。`InferShape`为const函数，不能修改Op的成员变量，参数为`const framework::InferShapeContext &ctx`，通过该参数可获取到输入输出以及属性。它的功能是：

  - 1). 做检查， 尽早报错：检查输入数据维度、类型等是否合法。
  - 2). 设置输出Tensor的形状。

通常`OpProtoMaker`和`Op`类的定义写在`.cc`文件中，和下面将要介绍的注册函数一起放在`.cc`中

### 定义OpKernel类

`MulKernel`继承自`framework::OpKernel`，带有下面两个模板参数:

- `typename DeviceContext`: 表示设备类型，不同设备(CPU、CUDA)共享同一个Kernel时，需加该模板参数，不共享则不加，一个不共享的例子是[`OnehotCrossEntropyOpKernel`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/cross_entropy_op.h#L43)。

- `typename T` : 表示数据类型，如`float`, `double`等。

需要为`MulKernel`类重写`Compute`接口。
- `Compute`接受一个输入参数：`const framework::ExecutionContext& context`。
- 与`InferShapeContext`相比，`ExecutionContext`增加了设备类型，同样可获取到输入输出和属性参数。
- `Compute`函数里实现`OpKernel`的具体计算逻辑。

下面是 `MulKernel` `Compute`的实现：

  ```cpp
  template <typename DeviceContext, typename T>
  class MulKernel : public framework::OpKernel {
  public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<Tensor>("X");
    auto* Y = context.Input<Tensor>("Y");
    auto* Z = context.Output<Tensor>("Out");
    Z->mutable_data<T>(context.GetPlace());
    auto& device_context = context.template device_context<DeviceContext>();
    math::matmul<DeviceContext, T>(*X, false, *Y, false, 1, Z, 0, device_context);
  }
  };
  ```

需要注意：**不同设备(CPU、CUDA)共享一个Op定义，是否则共享同一个`OpKernel`，取决于`Compute`调用的函数是否支持不同设备。**

`MulOp`的CPU、CUDA实现共享同一个`Kernel`。`OpKernel`不共享的例子可以参考：[`OnehotCrossEntropyOpKernel`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/cross_entropy_op.h#L43)。

为了使`OpKernel`的计算过程书写更加简单，并且CPU、CUDA的代码可以复用，我们通常借助 Eigen unsupported Tensor模块来实现`Compute`接口。关于在PaddlePaddle中如何使用Eigen库，请参考[使用文档](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/dev/use_eigen_cn.md)。

到此，前向Op实现完成。接下来，需要在`.cc`文件中注册该op和kernel。
反向Op类的定义，反向OpKernel的定义与前向Op类似，这里不再赘述。**但需注意反向Op没有`ProtoMaker`**。

### 注册Operator

- 在`.cc`文件中注册前向、反向Op类，注册CPU Kernel。

    ```cpp
    namespace ops = paddle::operators;
    REGISTER_OPERATOR(mul, ops::MulOp, ops::MulOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>)
    REGISTER_OPERATOR(mul_grad, ops::MulGradOp)
    REGISTER_OP_CPU_KERNEL(mul, ops::MulKernel<paddle::platform::CPUDeviceContext, float>);
    REGISTER_OP_CPU_KERNEL(mul_grad,
                  ops::MulGradKernel<paddle::platform::CPUDeviceContext, float>);
    ```

   在上面的代码中：

    - `REGISTER_OPERATOR` ： 注册`ops::MulOp`类，类型名为`mul`，该类的`ProtoMaker`为`ops::MulOpMaker`，注册`ops::MulOpGrad`，类型名为`mul_grad`。
    - `REGISTER_OP_CPU_KERNEL` ：注册`ops::MulKernel`类，并特化模板参数为`paddle::platform::CPUPlace`和`float`类型，同理，注册`ops::MulGradKernel`类。


- 在 `.cu`文件中注册CUDA Kernel。
    - 请注意，如果CUDA Kernel的实现基于Eigen unsupported模块，那么在 `.cu`的开始请加上宏定义 `#define EIGEN_USE_GPU`，代码示例如下：

    ```cpp
    // if use Eigen unsupported module before include head files
    #define EIGEN_USE_GPU

    namespace ops = paddle::operators;
    REGISTER_OP_CUDA_KERNEL(mul, ops::MulKernel<paddle::platform::CUDADeviceContext, float>);
    REGISTER_OP_CUDA_KERNEL(mul_grad,
                           ops::MulGradKernel<paddle::platform::CUDADeviceContext, float>);
    ```

### 编译

运行下面命令可以进行编译：

```
make mul_op
```

## 绑定Python

系统会对新增的op自动绑定Python，并链接到生成的lib库中。

## 实现单元测试

单测包括对比前向Op不同设备(CPU、CUDA)的实现、对比反向OP不同设备(CPU、CUDA)的实现、反向Op的梯度测试。下面介绍介绍[`MulOp`的单元测试](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_mul_op.py)。

### 前向Operator单测

Op单元测试继承自`OpTest`。各项更加具体的单元测试在`TestMulOp`里完成。测试Operator，需要：

1. 在`setUp`函数定义输入、输出，以及相关的属性参数。
2. 生成随机的输入数据。
3. 在Python脚本中实现与前向operator相同的计算逻辑，得到输出值，与operator前向计算的输出进行对比。
4. 反向计算已经自动集成进测试框架，直接调用相应接口即可。


  ```python
  import unittest
  import numpy as np
  from op_test import OpTest


  class TestMulOp(OpTest):
      def setUp(self):
          self.op_type = "mul"
          self.inputs = {
              'X': np.random.random((32, 84)).astype("float32"),
              'Y': np.random.random((84, 100)).astype("float32")
          }
          self.outputs = {'Out': np.dot(self.inputs['X'], self.inputs['Y'])}

      def test_check_output(self):
          self.check_output()

      def test_check_grad_normal(self):
          self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.5)

      def test_check_grad_ingore_x(self):
          self.check_grad(
              ['Y'], 'Out', max_relative_error=0.5, no_grad_set=set("X"))

      def test_check_grad_ingore_y(self):
          self.check_grad(
              ['X'], 'Out', max_relative_error=0.5, no_grad_set=set('Y'))
  ```

上面的代码首先导入依赖的包，下面是对`setUp`函数中操作的重要变量的详细解释：

- `self.op_type = "mul" ` : 定义类型，与operator注册时注册的类型一致。
- `self.inputs` : 定义输入，类型为`numpy.array`，并初始化。
- `self.outputs` : 定义输出，并在Python脚本中完成与operator同样的计算逻辑，返回Python端的计算结果。

### 反向operator单测

而反向测试中：
- `test_check_grad_normal`中调用`check_grad`使用数值法检测梯度正确性和稳定性。
  - 第一个参数`["X", "Y"]` : 指定对输入变量`X`、`Y`做梯度检测。
  - 第二个参数`"Out"` : 指定前向网络最终的输出目标变量`Out`。
  - 第三个参数`max_relative_error`：指定检测梯度时能容忍的最大错误值。
- `test_check_grad_ingore_x`和`test_check_grad_ingore_y`分支用来测试只需要计算一个输入梯度的情况。


### 编译和执行

`python/paddle/fluid/tests/unittests/` 目录下新增的 `test_*.py` 单元测试会被自动加入工程进行编译。

请注意，**不同于Op的编译测试，运行单元测试测时需要编译整个工程**，并且编译时需要打开`WITH_TESTING`, 即`cmake paddle_dir -DWITH_TESTING=ON`。编译成功后，执行下面的命令来运行单元测试：

```bash
make test ARGS="-R test_mul_op -V"
```

或者:

```bash
ctest -R test_mul_op
```

## 注意事项

- 注册Op时的类型名，需要和该Op的名字一样。即不允许在`A_op.cc`里面，注册`REGISTER_OPERATOR(B, ...)`等，这将会导致单元测试出错。
- 如果Op没有实现CUDA Kernel，请不要创建空的`*_op.cu`，这将会导致单元测试出错。
- 如果多个Op依赖一些共用的函数，可以创建非`*_op.*`格式的文件来存放，如`gather.h`文件。
