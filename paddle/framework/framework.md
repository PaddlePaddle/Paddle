# Design Doc: Framework

This design is after the learning of other deep learning systems and numerous group discussions.

## Operation and Operator

At the core of a deep learning system is the neural network.  A neural network is a directed graph of operations, where each operation is an instance of an *operator*, which is a C++ class with a `Run` method derived from the base class `Operator`. 

### `OperationProto` 

Users are supposed to describe neural networks by calling Python.  functions known as *operation creators*, which, in turn call a C++ function `paddle::framework::CreateOperation`.  This C++ function eases the adding of more language bindings.

Python and C++ have different function call syntax, so we define the parameter of `paddle::framework::CreateOperation` a protobuf message `OperationProto`.

### `OperatorProto`

We'd like to generate operator creators automatically from C++ code so to keep Python code always updated.  So we need to describe each C++ class in a protobuf message `OperatorProto`. We also need to fill in an `OperatorProto` message for each operator class and expose these messages to Python function `paddle.framework.create_operation_creators`.  We call this filling and exposing mechanism *operator registration*.


## Operators, Layers, Variables, Scope

### Operators as Functions

An operator is intrinsically a function, which has inputs and outputs.  For example, the functional representation of the GEMM operator is

```cpp
gemm(X, W, scale, act=LeRU) {
  unactivated = scale * X * W
  if !act {
    return unactivated
  }
  return unactivated, act(unactivated, cap)
}
```

Note that operators might call other operators.  In above example, `gemm` calls `act`.

### Gradient Operators

Each operator has a corresponding gradient operator that defines the gradient computation.

### Layers

If we describe a neural network by calling operator creators directly, our code would be lengthy.  It is easier to use layers creators, which, in addition to calling operator creators for the computation, creates model parameters.

For example, `paddle.layer.fc` and `paddle.layer.conv` both call `paddle.op.gemm`. They should also create and initialize `W`, the layer's parameter, and `act` and `scale`, the attributes.

### Variables

We prefer to represent inputs, outputs, and attributes in Variables, so an operation's output can be used to feed and to configure  other operations.

Variables should be able to hold the following types of values:

- int
- bool
- float
- half
- string
- tensor
- operator
- scope

Some Python API proposals:

```python
x = paddle.variable.new("x") # create a variable of not yet known type
x = paddle.variable.tensor("x") # create a tensor typed varible without value
x = paddle.variable.int("x", 10) # create an unnamed int varaible and set to 10
x = paddle.variable.operation("x", paddle.operator.gemm(...)) # an operation
x = paddle.variable.tensor("x", 
      numpy.random.randn(200, 100), # set the value of the tensor
      estimated = true)             # will be updated by the backward algorithm.
x.estimated = false                 # prevents from update.
```

Note that the variable has the following methods:

```cpp
class Variable {
 public:
  bool Estimated() const;
  bool SetEstimated(bool);
  template <typename T> const T& Get() const;
  template <typename T> T* GetMutable();
  template <typename T> bool IsType(T) const;
};
```

`Get` and `GetMutable` implements *lazy memory allocation*, as described in the [Variable design doc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/variable.md).

Note that `name` is not a property of a variable.  A variable can have various names in different scopes.

### Scope

In programming languages, variables belong to scopes.  Scopes enable the release of local variables when a stack frame pops.

A neural network is arguably equivalent to a program.  The *recurrent operator*, `paddle::operator::Recurrent`, is like `for` loop, and the conditional operator, `paddle::operator::Conditional`, is like `if/switch`.  They can have sub-networks as attributes, and `paddle::operator::Recurrent/Conditional::Run` creates a local scope before executing the sub-network.  At inference time, once `paddle::operator::Recurrent/Conditional::Run` frees the scope before completes.  At training time, it is the corresponding gradient operations' `paddle::operator::RecurrentGrad/ConditionalGrad::Run` who free the local scope.

The global and nested local scopes form a hierarchy.  The following Python functions make it convenient to program scopes:

1. `paddle.scope.current()` returns the current scope, which defaults to
1. `paddle.scope.global()`, which returns the top-level scope.

C++ code shouldn't maintain global status like the current scope to prevent unexpected inconsistancy.

```cpp
class Scope {
 public:
  Scope() : parent_(nullptr) {} // Constructor creates only global scopes. 
  
  Variable* FindVar(std::string name); // Find in the hierrachy or return null.
  Variable* CreateVar(StringPiece name, Variable*); // Find or create.
  
  Scope* CreateScope(StringPiece name); // Finds or creates a sub-scope.
  void DeleteScope(Scope*);             // Delete a sub-scope or raise exception
  
 private:
  std::map<std::string /*name*/, std::unique_ptr<Variable> > vars_;
  std::shared_ptr<Scope> parent_;
  std::vector<Scope*> children_;
  
  Mutex mutex_;  // Make this class thread-safe.
};
```

## Execution and Context

A neural network is a program.  Training or inference is to execute it.  The runtime environment of execution is known as a *context*:

1. a scope,
1. device(s), or places,
1. a flag indicates if we are training and should we create gradient operations and run backward

and can be defined as

```cpp
struct Context {
  Scope* scope_;
  std::vector<Place> places_; // a network might run on multiple devices.
  bool training_;
};
```

The Python API `paddle.train` can prepare a context before it calls C++ code `Operator::Run(const Context&)`.

As an example, `paddle::operator::Recurrent::Run(const Context& ctx)` can then create a new scope by calling `ctx.CreateScope`, and run the step-net with a new context around the new scope:

```cpp
class Recurrent {
 public:
  void Run(const Context& ctx) {
    auto subscope = ctx.scope_.CreateScope("");
    step_net_.Run(Context{subscope, ctx.places_, ctx.training_});
    if (!ctx.training_) {
      ctx.scope_.DeleteScope(subscope);
    }
};
```

Another example is that the Gemm operator need to create a tensor on `Context::places_[0]` and assign the tensor to its output variable:

```cpp
class Gemm {
 public:
  void Run(const Context& ctx) {
    if (paddle::platform::IsGPUPlace(ctx.places_[0])) {
      cuDNNGemm(
         Output(0).mutable_data<float>(ctx.places_[0], DerivedSizeFromInputs()),
         ...);
    } else {
      mkl::sgemm(
         Output(0).mutable_data<float>(ctx.places_[0], DerivedSizeFromInputs()),
         ...);
    }
  }
};
```

### Place

A place indicates a device and its type.  We have the following place definitions:

```cpp
struct GPUPlace {
  int device_; // GPU id.
};

struct CPUPlace {
  enum Type {
    X86,
    ARM5,
    ARM6,
    ...
  };
  Type type_;
};
```

We can add more Place implementations, like FPGAPlace and XeonPhiPlace, in the future.

### Gradient Operators

A gradient operator should be build and linked only if we are building a binary that supports training.  If we are building an "inference-only" binary, we shouldn't link gradient operators.

Gradient operations should be created only if we are going to train a neural network.
