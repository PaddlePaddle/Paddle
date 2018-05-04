# How to write a new operator

 - [Background](#background)
 - [Implementing C++ Types](#implementing-c-types)
   - [Defining ProtoMaker](#defining-protomaker)
   - [Defining Operator](#defining-operator)
   - [Defining OpKernel](#defining-opkernel)
   - [Registering Operator and OpKernel](#registering-operator-and-opkernel)
   - [Compilation](#compilation)
 - [Python Binding](#python-binding)
 - [Unit Tests](#unit-tests)
   - [Testing Forward Operators](#testing-forward-operators)
   - [Testing Backward Operators](#testing-backward-operators)
   - [Compiling and Running](#compiling-and-running)
 - [Remarks](#remarks)
## Background

Here are the base types needed. For details, please refer to the design docs.

- `class OpProtoAndCheckerMaker`: Describes an Operator's input, output, attributes and description, mainly used to interface with Python API.
- `framework::OperatorBase`: Operator (Op)base class.
- `framework::OpKernel`: Base class for Op computation kernel.
- `framework::OperatorWithKernel`: Inherited from OperatorBase, describing an operator with computation kernels.


Operators can be categorized into two groups: operator with kernel(s) and operator without kernel(s). An operator with kernel(s) inherits from `OperatorWithKernel` while the one without kernel(s) inherits from `OperatorBase`. This tutorial focuses on implementing operators with kernels. In short, an operator includes the following information:


<table>
<thead>
<tr>
<th>Information</th>
<th> Where is it defined</th>
</tr>
</thead>
<tbody>
<tr>
<td>OpProtoMake definition </td>
<td> `.cc`files, Backward Op does not need an OpProtoMake interface. </td>
</tr>
<tr>
<td>Op definition  </td>
<td> `.cc` files</td>
</tr>
<tr>
<td>Kernel implementation  </td>
<td> The kernel methods shared between CPU and CUDA are defined in `.h` files. CPU-specific kernels live in `.cc` files, while CUDA-specific kernels are implemented in `.cu`files.</td>
</tr>
<tr>
<td>Registering the Op  </td>
<td> Ops are registered in `.cc` files; For Kernel registration, `.cc` files contain the CPU implementation, while `.cu` files contain the CUDA implementation.</td>
</tr>
</tbody>
</table>


New Operator implementations are added to the list [paddle/operators](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/operators), with file names in the format `*_op.h` (if applicable), `*_op.cc`, `*_op.cu` (if applicable).** The system will use the naming scheme to automatically build operators and their corresponding Python extensions.**


Let's take matrix multiplication operator, [MulOp](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/mul_op.cc), as an example to introduce the writing of an Operator with Kernel.


## Implementing C++ Types


### Defining ProtoMaker

Matrix Multiplication can be written as $Out = X * Y$, meaning that the operation consists of two inputs and pne output.

First, define `ProtoMaker` to describe the Operator's input, output, and additional comments:

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

[`MulOpMaker`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/mul_op.cc#L76-L127)is inherited from`framework::OpProtoAndCheckerMaker`, consisting of 2 variables in the constructor：

   - `framework::OpProto` stores Operator input and variable attribute, used for generating Python API interfaces.
   - `framework::OpAttrChecker` is used to validate variable attributes.

The constructor utilizes `AddInput`, `AddOutput`, and `AddComment`, so that the corresponding information will be added to `OpProto`.

The code above adds two inputs `X` and `Y` to `MulOp`, an output `Out`, and their corresponding descriptions, in accordance to Paddle's [naming convention](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/dev/name_convention.md).


An additional example [`ScaleOp`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/scale_op.cc#L38-L55) is implemented as follows:

```cpp
template <typename AttrType>
class ScaleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ScaleOpMaker(OpProto *proto, OpAttrChecker *op_checker)
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

Note `AddAttr<AttrType>("scale", "...").SetDefault(1.0);` adds `scale`constant as an attribute, and sets the default value to 1.0.


### Defining Operator

The following code defines the interface for MulOp:

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

[`MulOp`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/mul_op.cc#L24) is inherited from `OperatorWithKernel`. Its `public` member

```cpp
using framework::OperatorWithKernel::OperatorWithKernel;
```

expresses an operator constructor using base class `OperatorWithKernel`, alternatively written as

```cpp
MulOp(const std::string &type, const framework::VariableNameMap &inputs,
      const framework::VariableNameMap &outputs,
      const framework::AttributeMap &attrs)
  : OperatorWithKernel(type, inputs, outputs, attrs) {}
```

`InferShape` interface needs to be re-written.`InferShape` is a constant method and cannot modify Op's member variables, its constant member `const framework::InferShapeContext &ctx` can be used to extract input, output, and attributes. It functions to

  - 1). validate and error out early: it checks input data dimensions and types.
  - 2). configures the tensor shape in the output.

Usually `OpProtoMaker` and `Op`'s type definitions are written in `.cc` files, which also include the registration methods introduced later.

### Defining OpKernel

`MulKernel` inherits `framework::OpKernel`, which includes the following templates:

- `typename  DeviceContext` denotes device context type. When different devices, namely the CPUDeviceContext and the CUDADeviceContext, share the same kernel, this template needs to be added. If they don't share kernels, this must not be added. An example of a non-sharing kernel is [`OnehotCrossEntropyOpKernel`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/cross_entropy_op.h#L43).

- `typename T` denotes data type, such as `float` or `double`.

`MulKernel` types need to rewrite the interface for `Compute`.

- `Compute` takes one input parameter: `const framework::ExecutionContext& context`.
- Compared with `InferShapeContext`, `ExecutionContext` includes device types, and can similarly extract input, output, and attribute variables.
- `Compute` implements the computation logics of an `OpKernel`.

`MulKernel`'s implementation of `Compute` is as follows:

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

Note that **different devices (CPU, CUDA)share one Op definition; whether or not they share the same `OpKernel` depends on whether `Compute` calls functions can support both devices.**

`MulOp`'s CPU and CUDA share the same `Kernel`. A non-sharing  `OpKernel` example can be seen in [`OnehotCrossEntropyOpKernel`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/cross_entropy_op.cc).

To ease the writing of `OpKernel` compute, and for reusing code cross-device, [`Eigen-unsupported Tensor`](https://bitbucket.org/eigen/eigen/src/default/unsupported/Eigen/CXX11/src/Tensor/README.md?fileviewer=file-view-default) module is used to implement `Compute` interface. To learn about how the Eigen library is used in PaddlePaddle, please see [usage document](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/dev/use_eigen_en.md).


This concludes the forward implementation of an operator. Next its operation and kernel need to be registered in a `.cc` file.

The definition of its corresponding backward operator, if applicable, is similar to that of an forward operator. **Note that a backward operator does not include a `ProtoMaker`**.

### Registering Operator and OpKernel

- In `.cc` files, register forward and backward operator classes and the CPU kernel.

    ```cpp
    namespace ops = paddle::operators;
    REGISTER_OPERATOR(mul, ops::MulOp, ops::MulOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>)
    REGISTER_OPERATOR(mul_grad, ops::MulGradOp)

    REGISTER_OP_CPU_KERNEL(mul, ops::MulKernel<paddle::platform::CPUDeviceContext, float>);
    REGISTER_OP_CPU_KERNEL(mul_grad,
                  ops::MulGradKernel<paddle::platform::CPUDeviceContext, float>);
    ```

   In that code block,

    - `REGISTER_OPERATOR` registers the `ops::MulOp` class, type named `mul`, its type `ProtoMaker` is `ops::MulOpMaker`, registering `ops::MulOpGrad` as `mul_grad`.
    - `REGISTER_OP_WITHOUT_GRADIENT` registers an operator without gradient.
    - `REGISTER_OP_CPU_KERNEL` registers `ops::MulKernel` class and specialized template types `paddle::platform::CPUPlace` and `float`, which also registers `ops::MulGradKernel`.


- Registering CUDA Kernel in `.cu` files
    - Note that if CUDA Kernel is implemented using the `Eigen unsupported` module, then on top of `.cu`, a macro definition `#define EIGEN_USE_GPU` is needed, such as

    ```cpp
    // if use Eigen unsupported module before include head files
    #define EIGEN_USE_GPU

    namespace ops = paddle::operators;
    REGISTER_OP_CUDA_KERNEL(mul, ops::MulKernel<paddle::platform::CUDADeviceContext, float>);
    REGISTER_OP_CUDA_KERNEL(mul_grad,
                           ops::MulGradKernel<paddle::platform::CUDADeviceContext, float>);
    ```

### Compilation

Run the following commands to compile.

```
# maybe you need to rerun cmake
make mul_op
```

## Python Binding

The system will automatically bind to Python and link it to a generated library.

## Unit Tests

Unit tests for an operator include

1. comparing a forward operator's implementations on different devices,

2. comparing a backward operator's implementation on different devices, and

3. a scaling test for the backward operator.

Here, we introduce the [unit tests for `MulOp`](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/unittests/test_mul_op.py).

### Testing Forward Operators

A forward operator unit test inherits `unittest.TestCase` and defines metaclass `__metaclass__ = OpTestMeta`. More concrete tests are performed in `OpTestMeta`. Testing a forward operator requires the following:

1. Defining input, output and relevant attributes in `setUp` method.

2. Generating random input data.

3. Implementing the same computation logic in a Python script.

4. Call check gradient function to check the backward operator.

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
Get its output, and compare it with the forward operator's own output.

The code above first loads required packages. In addition, we have

- `self.op_type = "mul" ` defines the type that is identical to what the operator's registered type.
- `self.inputs` defines input, with type `numpy.array` and initializes it.
- `self.outputs` defines output and completes the same operator computation in the Python script, and returns its result from the Python script.

### Testing Backward Operators

Some key points in checking gradient above include:

- `test_normal` calls `check_grad` to validate scaling tests' correctness and stability through numeric methods.
  - The first variable `["X", "Y"]` appoints `X` and `Y` to be scale tested.
  - The second variable `"Out"` points to the network's final output target `Out`.
  - The third variable `max_relative_error` points to the maximum relative tolerance error during scaling tests.
- `test_check_grad_ingore_x` and `test_check_grad_ingore_y`branches test the cases where there is only one scaling input.

### Compiling and Running


Any new unit testing file of the format `test_*.py`  added to the director `python/paddle/fluid/tests/unittests/` is automatically added to the project to compile.

Note that **unlike the compile test for Ops, running unit tests requires compiling the entire project** and requires compiling with flag `WITH_TESTING` on i.e. `cmake paddle_dir -DWITH_TESTING=ON`.

After successfully compiling the project, run the following command to run unit tests:

```bash
make test ARGS="-R test_mul_op -V"
```

Or,

```bash
ctest -R test_mul_op
```

## Remarks

- The type with which an operator is registered needs to be identical to the Op's name. Registering `REGISTER_OPERATOR(B, ...)` in `A_op.cc` will cause unit testing failures.
- If the operator does not implement a CUDA kernel, please refrain from creating an empty `*_op.cu` file, or else unit tests will fail.
- If multiple operators rely on some shared methods, a file NOT named `*_op.*` can be created to store them, such as `gather.h`.
