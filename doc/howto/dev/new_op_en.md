# How to write a new operator

 - [Background](#Background)
 - [Implementing C++ Types](#Implementing_C++_Types)
   - [Defining ProtoMaker](#Defining_ProtoMaker)
   - [Defining Operator](#Defining_Operator)


## Background

Here are the base types needed. For details, please refer to the design docs.

- `framework::OperatorBase`: Operator (Op)base class.
- `framework::OpKernel`: Base class for Op computation.
- `framework::OperatorWithKernel`: Inherited from OperatorBase, describing an operator with computation.
- `class OpProtoAndCheckerMaker`: Describes an Operator's input, output, attributes and description, mainly used to interface with Python API.

An operator can be differentiated by whether in has kernel methods. An operator with kernel inherits from `OperatorWithKernel` while the ones without inherit from `OperatorBase`. This tutorial focuses on implementing operators with kernels. In short, an operator includes the following information:


 Information           | Where is it defined
--------------  | :----------------------
OpProtoMake definition  | `.cc`files, Backward Op does not need an OpProtoMake interface.
Op definition           | `.cc` files
Kernel implementation       | The kernel methods shared between CPU and GPU are defined in `.h` files. CPU-specific kernels live in `.cc` files, while GPU-specific kernels are implemented in `.cu`files.
Registering the Op           | Ops are registered in `.cc` files; For Kernel registration, `.cc` files contain the CPU implementation, while `.cu` files contain the GPU implementation.


New Operator implementations are added to the list [paddle/operators](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/operators), with file names in the format `*_op.h` (if applicable), `*_op.cc`, `*_op.cu` (if applicable).** The system will use the naming scheme to automatically build operators and their corresponding Python extensions. **


Let's take matrix multiplication operator, [MulOp](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/mul_op.cc), as an example to introduce the writing of an Operator with Kernel.


## Implementing C++ Types


### 1. Defining Class ProtoMaker

Matrix Multiplication can be written as $Out = X * Y$, meaning that the operation consists of two inputs and pne output.

First, define `ProtoMaker` to describe the Operator's input, output, and additional comments:

```cpp
class MulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MulOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
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

[`MulOpMaker`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/mul_op.cc#L43)is inherited from`framework::OpProtoAndCheckerMaker`, consisting of 2 variables in the constructor：

   - `framework::OpProto` stores Operator input and variable attribute, used for generating Python API interfaces.
   - `framework::OpAttrChecker` is used to validate variable attributes.

The constructor utilizes `AddInput`, `AddOutput`, and `AddComment`, so that the corresponding information will be added to `OpProto`.

The code above adds two inputs `X` and `Y` to `MulOp`, an output `Out`, and their corresponding descriptions, in accordance to Paddle's [naming convention](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/name_convention.md).


An additional example [`ScaleOp`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/scale_op.cc#L37) is implemented as follows:

```cpp
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

There are two changes in this example:

- `AddInput("X","...").NotInGradient()` expresses that input `X` is not involved in `ScaleOp`'s corresponding computation. If an input to an operator is not participating in back-propagation, please explicitly set `.NotInGradient()`.

- `AddAttr<AttrType>("scale", "...").SetDefault(1.0);`  adds `scale`constant as an attribute, and sets the default value to 1.0.


### 2. Defining Operator

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

[`MulOp`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/mul_op.cc#L22) is inherited from `OperatorWithKernel`. Its `public` member

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
