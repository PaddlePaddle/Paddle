# Design Doc: Functions

Following the overall [design](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/functions_operators_layers.md), this document is about the functions and how they compose into operators.


## Functions operate on scalar values

Basically, to the ease of definitions, functions operate on scalar values. But most operators operate on Tensors. So we use `transform` function invoke on each `scalar` values of Tensors. For example:

```cpp
// elemwise sigmoid
template <typename T>
__host__ __device__ T Sigmoid(T a) {
  // tanh is defined in CUDA.
  return (tanh(a) + 1) / 2;
}

template <typename Place>
class SigmoidOpKernel : public framework::OpKernel {};

template <>
class SigmoidOpKernel<CPUPlace> : public framework::OpKernel {
 public:
  void Compute(ctx) {
    std::transform(
        ctx.Input<Tensor>("X")->data(),
        ctx.Input<Tensor>("X")->data() + product(ctx.Input<Tensor>("X")->dims()),
        ctx.Output<Tensor>("Out")->mutable_data(),
        Sigmoid);
  }
};

template <>
class SigmoidOpKernel<GPUPlace> : public framework::OpKernel {
public:
  void Compute(ctx) {
    thrust::transform(
        ctx.Input<Tensor>("X")->data(),
        ctx.Input<Tensor>("X")->data() + product(ctx.Input<Tensor>("X")->dims()),
        ctx.Output<Tensor>("Out")->mutable_data(),
        Sigmoid);
  }
};
```

The `thust::tranform` and `std::transform` share the same interface. We partial specialize `SigmoidOpKernel` to use `thrust` on GPU and `std` on CPU. The developer just need to define a function operates on scalar values, i.e., the `sigmoid` function.

To mark `sigmoid` function as `__host__ __device__` could make that function can be execute inside a GPU kernel function(which marked as `__global__`) and on CPU.


## Use functor for passing operator's attribute

A functor is a C++ class who overloads the `operator()`. A functor instance acts like a function. However, a functor can bind parameters while constructing. To use functor, we can let operator's attribute pass to GPU kernel.

For example,

```cpp
template <typename T>
class Scale {
 public:
  Scale(T scale): scale_(scale) {}
  
  __device__ __host__ T operator()(T x) const {
    return x * scale_;
  }
 private:
  T scale_;
};

...

template <typename T>
class ScaleOpKernel<CPUPlace> : public framework::OpKernel {
 public:
  void Compute(ctx) {
    std::transform(
        ctx.Input<Tensor>("X")->data(),
        ctx.Input<Tensor>("X")->data() + product(ctx.Input<Tensor>("X")->dims()),
        ctx.Output<Tensor>("Out")->mutable_data(),
        Scale<T>(ctx.Attr<T>("scale")));  // create a instance of Scale functor.
  }
};
...
```

## Fuse kernels by function calls

It is easy to fuse GPU kernels if we define all functions as `__device__ __host__` method. To invoke a function inside `__device__ __host__` will be performed in a single GPU kernel.

For example,

```cpp
template <typename T>
__device__ __host__ T Sigmoid(T x) { ... }

template <typename T>
__device__ __host__ T Dropout(T x, float rate) { ... }

template <typename T>
__device__ __host__ T SigmoidWithDropout(T x, float rate) {
  T tmp = Sigmoid(x);
  return Dropout(tmp, rate);
}
```

The `SigmoidWithDropout` will be performed in a single GPU kernel. A functor can invoke functions as well. It could be useful that a function could be shared in two operators, one operator has a attribute, the other does not. For example,

```cpp
template <typename T1, typename T2>
__device__ __host__ T Multiply(T1 x, T2 y) { return x * y; }

template <typename T, typename S>
class Scale {
 public:
  Scale(S scale): scale_{scale} {}
  __device__ __host__ T operator(T x) {
    return Multiply(x, scale_);
  }
  
 private:
  S scale_;
};

...
template <typename T>
class ElemwiseMulOpKernel<CPUPlace> : public framework::OpKernel {
 public:
  void Compute(ctx) {
    std::transform(
        ctx.Input<Tensor>("X")->data(),
        ctx.Input<Tensor>("X")->data() + product(ctx.Input<Tensor>("X")->dims()),
        ctx.Input<Tensor>("Y")->data()
        ctx.Output<Tensor>("Out")->mutable_data(),
        Multiply);
   }
};

...

template <typename T>
class ScaleOpKernel<CPU> : public framework::OpKernel {
  void Compute(ctx) {
    std::transform(
        ctx.Input<Tensor>("X")->data(),
        ctx.Input<Tensor>("X")->data() + product(ctx.Input<Tensor>("X")->dims()),
        ctx.Output<Tensor>("Out")->mutable_data(),
        Scale<T>(ctx.Attr<T>("scale")));  // create a instance of Scale functor.
  }
};
```

The `ScaleOp` takes `scale` attribute, but elemwise multiply operator does not take any attribute. However, they share the same `Multiply` function.

## Pre-defined Operators, OpProtoMakers and OpKernels

Since the kernels of operator are seperate with operators and operator protobuf makers, we can share the same `Operator` and `OpProtoMakers` for same kinds of operator. Let us use `UnaryOperator` as an example.

```cpp
class UnaryOperator : public framework::OpWithKernels {
public:
  void InferShape(ctx) {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    out->Resize(x->ddims());
  }
};

REGISTER_OP(sigmoid, UnaryOperator, ...);
```

The `OpProtoMakers` is a little bit difficult, because every has a different comment. However, we can still make a base class for `UnaryOpProtoMakers`.

```cpp
class UnaryOpProtoMakerBase : public framework::OpProtoAndCheckerMaker {
 public:
  UnaryOpProtoMakerBase(...) {
    AddInput("X", "the input tensor");
    AddOutput("Out", "the output tensor");
  }
};

class SigmoidOpProtoMaker : public UnaryOpProtoMakerBase {
 public:
  SigmoidOpProtoMaker(...) {
    AddComment("Sigmoid operator. The equation is `(Out = tanh(X) + 1)/ 2`.")
  }
};

REGISTER_OP(sigmoid, UnaryOperator, SigmoidOpProtoMaker, ...);
```

The `OpKernels` for unary operator are basically the same. We could extract a class for `UnaryKernel`. It likes

```cpp
template <typename Functor>
class UnaryKernel<CPUPlace()> : public OpKernel {
 public:
  void Compute(ctx) const {
    auto* in = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    std::transform(
        in->data(),
        in->data() + product(ctx.Input<Tensor>("X")->dims()),
        out->mutable_data(),
        Functor(ctx));
  }
};
```

All functors which are registered to UnaryKernel must have a constructor with context as parameter, so functor can get attribute from the context.
