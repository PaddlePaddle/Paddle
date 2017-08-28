# 如何写新的Operator

 - [概念简介](#概念简介)
 - [实现C++类](#实现C++类)
   - [定义ProtoMaker类](#定义ProtoMaker类)
   - [定义Operator类](#定义Operator类)
   - [定义OpKernel类](#定义OpKernel类)
   - [注册类](#注册类)
   - [编译](#编译)
 - [绑定Python](#绑定Python)
 - [实现单元测试](#实现单元测试)
   - [前向Operator单测](#前向Operator单测)
   - [反向Operator单测](#反向Operator单测)


## 概念简介

简单介绍需要用到基类，详细介绍请参考设计文档。

- `framework::OperatorBase`: Operator(简写，Op)基类。
- `framework::OpKernel`: Op计算函数的基类，称作Kernel。
- `framework::OperatorWithKernel`：继承自OperatorBase，Op有计算函数，称作有Kernel。
- `class OpProtoAndCheckerMaker`：描述该Op的输入、输出、属性、注释,主要用于Python API接口生成

依据是否包含kernel，将Op分为两种：包含Kernel的Op和不包含kernel的Op，前者Op的定义继承自`OperatorBase`，后者继承自`OperatorWithKernel`。本教程主要介绍带Kernel的Op如何写，简单总结如下：

Forward Op需要包含：

   - OpProtoMake定义
   - Op定义
   - Kernel实现
     
与之对应的Backward Op包含：

   - Op定义
   - Kernel实现

下面以矩阵乘操作，即[MulOp](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/mul_op.cc)为例来介绍如何写带Kernel的Operator。


## 实现C++类


### 1. 定义ProtoMaker类

矩阵乘的公式：$Out = X * Y$, 可见该计算由两个输入，一个输出组成。首先定义`ProtoMaker`来描述该Op的输入、输出及注释：
    
```
class MulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MulOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The first input of mul op");
    AddInput("Y", "The second input of mul op");
    AddOutput("Out", "The output of mul op");
    AddComment(R"DOC(
Two Element Mul Operator.
The equation is: Out = X * Y
)DOC");
  }
};
```
   
[`MulOpMaker`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/mul_op.cc#L43)继承自`framework::OpProtoAndCheckerMaker`，构造函数包括2个：

   - `framework::OpProto` ： 前者存储Op的输入输出和参数属性，将用于Python API接口的生成。
   - `framework::OpAttrChecker` ：后者用于检查参数属性的合法性。
   
构造函数里通过`AddInput`添加输入参数，通过`AddOutput`添加输出参数，通过`AddComment`添加该Op的注释，这些函数会将对应内容添加到`OpProto`中。

在`MulOp`中添加两个输入`X`和`Y`，添加了一个输出`Out`，并解释了各自含义，该命名尽可能的规范。

   
再举个[`ScaleOp`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/scale_op.cc#L37)的例子：
   
```
template <typename AttrType>
class ScaleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ScaleOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input tensor of scale operator.").NotInGradient();
    AddOutput("Out", "The output tensor of scale operator.").NotInGradient();
    AddComment(R"DOC(Scale operator
The equation is: Out = scale*X
)DOC");
    AddAttr<AttrType>("scale", "scale of scale operator.").SetDefault(1.0);
  }
};
```
 
 在这个例子里，两处不同：
 
  - `AddInput("X","...").NotInGradient()` : 表示`X`这个输入不参与`ScaleOp`对应的梯度Op计算之中。
  - `AddAttr<AttrType>("scale", "...").SetDefault(1.0);` : 增加`scale`系数，作为参数属性，并且设置默认值为1.0。
   

### 2. 定义Operator类


```c++
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

[`MulOp`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/mul_op.cc#L22)继承自`OperatorWithKernel`。`public`成员：
	 
```c++
using framework::OperatorWithKernel::OperatorWithKernel;
```

这句表示使用基类`OperatorWithKernel`的构造函数，也可写成：
   
```c++
MulOp(const std::string &type, const framework::VariableNameMap &inputs,
      const framework::VariableNameMap &outputs,
      const framework::AttributeMap &attrs)
  : OperatorWithKernel(type, inputs, outputs, attrs) {}
```	
	
还需要重写`InferShape`接口。`InferShape`为const函数，不能修改Op的成员变量，参数为`const framework::InferShapeContext &ctx`，通过该参数可获取到输入输出以及属性。它的功能是：
	 - 1). 做检查， 尽早报错：检查输入数据维度、类型等是否合法
	 - 2). 设置输出Tensor的形状

通常`OpProtoMaker`和`Op`类的定义写在`.cc`文件中，和要讲到的注册函数一起放在`.cc`中

### 3. 定义OpKernel类

```C++
template <typename Place, typename T>
class MulKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<Tensor>("X");
    auto* Y = context.Input<Tensor>("Y");
    auto* Z = context.Output<Tensor>("Out");
    Z->mutable_data<T>(context.GetPlace());
    auto* device_context =
        const_cast<platform::DeviceContext*>(context.device_context_);
    math::matmul<Place, T>(*X, false, *Y, false, 1, Z, 0, device_context);
  }
};
```

`MulKernel`继承自`framework::OpKernel`，带有模板参数:

  - `typename  Place`: 表示设备类型，不同设备(CPU、GPU)共享同一个Kernel时，需加该模板参数，不共享则不加，一个不共享的例子是[`OnehotCrossEntropyOpKernel`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/cross_entropy_op.h#L43)。
  
 - `typename T` : 表示数据类型，如`float`, `double`等。
   
`MulKernel`需要重写`Compute`接口，该接口参数为`const framework::ExecutionContext& context`, `ExecutionContext`相比`InferShapeContext`增加了设备类型，同样可获取到输入输出和属性参数，`Compute`函数里写具体实现时。
   
注意，不同设备(CPU、GPU)共享一个Op定义，是否则共享同一个`OpKernel`，取决于`Compute`调用的函数是否支持不同设备。`MulOp`的CPU、GPU实现共享同一个`Kernel`，`OpKernel`不共享的例子可以参考[`OnehotCrossEntropyOpKernel`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/cross_entropy_op.h#L43)。 
   
到此前向Op实现完成，需要在`.cc`文件中注册该op和kernel。反向Op类的定义和Kernel定义与前向Op类似，这里不再重复。但注意，反向Op没有`ProtoMaker`。
   
### 4. 注册类

在`.cc`文件中注册前向、反向Op类，注册CPU Kernel。

```c++
namespace ops = paddle::operators;
REGISTER_OP(mul, ops::MulOp, ops::MulOpMaker, mul_grad, ops::MulOpGrad);
REGISTER_OP_CPU_KERNEL(mul, ops::MulKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(mul_grad,
              ops::MulGradKernel<paddle::platform::CPUPlace, float>);
```
    
  - `REGISTER_OP` ： 注册`ops::MulOp`类，类型名为`mul`，该类的`ProtoMaker`为`ops::MulOpMaker`，注册`ops::MulOpGrad`，类型名为`mul_grad`，
  - `REGISTER_OP_WITHOUT_GRADIENT` ： 用于注册没有反向的Op。
  - `REGISTER_OP_CPU_KERNEL` ：注册`ops::MulKernel`类，并特化模板参数为`paddle::platform::CPUPlace`和`float`类型，同理，注册`ops::MulKernel`类。

在 `.cu`文件中注册GPU Kernel。
   
```c++
namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(mul, ops::MulKernel<paddle::platform::GPUPlace, float>);
REGISTER_OP_GPU_KERNEL(mul_grad,
                       ops::MulGradKernel<paddle::platform::GPUPlace, float>);
```

### 5. 编译

在[paddle/operators/CMakeLists.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/CMakeLists.txt)文件中添加编译。
   
```
op_library(mul_op SRCS mul_op.cc mul_op.cu DEPS math_function)
```
   
下面命令可以编译：
   
```
make mul_op
```

## 绑定Python

- 绑定Python 
 
    在 [`paddle/pybind/pybind.cc 
`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/pybind/pybind.cc)文件中添加该类：

    ```
    USE_OP(mul);
    ```
    如果只实现了CPU版本，则使用`USE_CPU_ONLY_OP`:
    
    ```
    USE_CPU_ONLY_OP(gather);
    ```
    
    使用`USE_OP`告知编译器需要链接该Op的目标文件，具体解释参考[代码注释](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/op_registry.h#L81)。
    
    
 - 生成库

   在 [`paddle/pybind/CMakeLists.txt`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/pybind/CMakeLists.txt)文件添加类到`DEPS`中，使得该Op可以链接到生成的lib库中。
   
   ```
   if(WITH_PYTHON)
     cc_library(paddle_pybind SHARED
     SRCS pybind.cc
     DEPS pybind python backward
     mul_op
     minus_op)
   endif(WITH_PYTHON)
   ```

## 实现单元测试

单测包括对比前向Op不同设备(CPU、GPU)的实现、对比反向OP不同设备(CPU、GPU)的实现、反向Op的梯度测试。下面介绍介绍[`MulOp`的单测](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/framework/tests/test_mul_op.py)。

### 前向Operator单测

前向Op单测继承自`unittest.TestCase`，并定义元类`__metaclass__ = OpTestMeta`，具体单测流程在`OpTestMeta`里完成。需在`setUp`函数定义输入输出和属性参数，以及Python对比的输出值。

```
import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta

class TestMulOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "mul"
        self.inputs = {
            'X': np.random.random((32, 84)).astype("float32"),
            'Y': np.random.random((84, 100)).astype("float32")
        }
        self.outputs = {'Out': np.dot(self.inputs['X'], self.inputs['Y'])}
```
   首先需要`import`必要的包,下面详细解释其他值：
   
   - `self.type = "mul" ` : 定义类型，和注册的类型一致。
   - `self.inputs` : 定义输入，类型为Numpy.array，并初始化。
   - `self.outputs` : 定义输出，并得到Python结算结果。

 
### 反向Operator单测

反向Op单测继承自`GradientChecker`，而`GradientChecker`集成自`unittest.TestCase`，所以反向单测函数需要`test_`开头。

 ```
 class MulGradOpTest(GradientChecker):
    def test_mul(self):
        op = create_op("mul")
        inputs = {
            'X': np.random.random((32, 84)).astype("float32"),
            'Y': np.random.random((84, 100)).astype("float32")
        }
        self.compare_grad(op, inputs)      
        # mul op will enlarge the relative error
        self.check_grad(
            op, inputs, set(["X", "Y"]), "Out", max_relative_error=0.5)
 ```

   - 调用`create_op("mul")`创建反向Op对应的前向Op。
   - 定义输入`inputs`。
   - 调用`compare_grad`函数对比CPU、GPU计算结果。
   - 调用`check_grad`检查梯度稳定性。
