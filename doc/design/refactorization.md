# Design Doc: Refactorization Overview

The goal of refactorizaiton include:

1. Make it easy for external contributors to write new elementory computaiton operations.
1. Make the codebase clean and readable.
1. Introduce a new design of computation representation -- a computation graph of operators and variables.
1. The graph representation helps implementing auto-scalable and auto fault recoverable distributed computing.

## Computation Graphs

1. PaddlePaddle represent the computation, training and inference of DL models, by computation graphs.

  1. Please dig into [computation graphs](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/graph.md) for a solid example.

1. Users write Python programs to describe the graphs and run it (locally or remotely).

1. A graph is composed of *variables* and *operators*.

1. The description of graphs must be able to be serialized/deserialized, so it

   1. could to be sent to the cloud for distributed execution, and
   1. be sent to clients for mobile or enterprise deployment.

1. The Python program do

   1. *compilation*: runs a Python program to generate a protobuf message representation of the graph and send it to
      1. the C++ library `libpaddle.so` for local execution,
      1. the master process of a distributed training job for training, or
      1. the server process of a Kubernetes serving job for distributed serving.
   1. *execution*: according to the protobuf message, constructs instances of class `Variable` and `OperatorBase`, and run them.

## Description and Realization

At compile time, the Python program generates protobuf message representation of the graph, or the description of the graph.

At runtime, the C++ program realizes the graph and run it.

| | Representation (protobuf messages) | Realization (C++ class objects) |
|---|---|---|
|Data|[VarDesc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/framework.proto#L107)|[Variable](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/variable.h#L24)|
|Operation|[OpDesc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/framework.proto#L35)|[Operator](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/operator.h#L64)|
|Block|BlockDesc|Block|

The word *graph* is exchangable with *block* in this document.  A graph represent computation steps and local variables as a C++/Java program block, or a pair of { and }.

## Compilation and Execution

1. Run an applicaton Python program to describe the graph.  In particular,

   1. create VarDesc to represent local/intermediate variables,
   1. create operators and set attributes,
   1. validate attribute values,
   1. inference the type and the shape of variables,
   1. plan for memory-reuse for variables,
   1. generate backward and optimization part of the Graph.
   1. possiblly split the graph for distributed training.

1. The invocation of `train` or `infer` in the application Python program:

   1. create a new Scope instance in the [scope hierarchy](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/scope.md) for each run of a block,
      1. realize local variables defined in the BlockDesc message in the new scope,
      1. a scope is similar to the stack frame in programming languages,

   1. create an instance of class `Block`, in which,
      1. realize operators in the BlockDesc message,

   1. run the Block by calling
      1. `Block::Eval(vector<Variable>* targets)` for forward and backward computations, or
      1. `Block::Eval(vector<Operator>* targets)` for optimization.


## Intermediate Representation (IR)

```text
Compile Time -> IR -> Runtime
```

### Benefit

- Optimization
  ```text
  Compile Time -> IR -> Optimized IR -> Runtime
  ```
- Send automatically partitioned IR to different nodes.
  - Automatic data parallel
    ```text
    Compile Time
    |-> Single GPU IR
        |-> [trainer-IR-0, trainer-IR-1, pserver-IR]
            |-> Node-0 (runs trainer-IR-0)
            |-> Node-1 (runs trainer-IR-1)
            |-> Node-2 (runs pserver-IR)
    ```
  - Automatic model parallel (planned for future)

---

# Operator/OpWithKernel/OpKernel

![class_diagram](http://api.paddlepaddle.org/graphviz?dot=https://gist.githubusercontent.com/reyoung/53df507f6749762675dff3e7ce53372f/raw/49caf1fb70820fb4a6c217634317c9306f361f36/op_op_with_kern_class_diagram.dot)

---

# Operator
![class_diagram](http://api.paddlepaddle.org/graphviz?dot=https://gist.githubusercontent.com/reyoung/53df507f6749762675dff3e7ce53372f/raw/dd598e8f1976f5759f58af5e5ef94738a6b2e661/op.dot)

* `Operator` is the fundamental building block as the user interface.
    * Operator stores input/output variable name, and attributes.
    * The `InferShape` interface is used to infer output variable shapes by its input shapes.
    * Use `Run` to compute `input variables` to `output variables`.

---

# OpWithKernel/Kernel

![class_diagram](http://api.paddlepaddle.org/graphviz?dot=https://gist.githubusercontent.com/reyoung/53df507f6749762675dff3e7ce53372f/raw/9d7f4eba185cf41c8e2fbfb40ae21890dbddcd39/op_with_kernel.dot)

* `OpWithKernel` inherits `Operator`.
* `OpWithKernel` contains a Kernel map.
    * `OpWithKernel::Run` get device's kernel, and invoke `OpKernel::Compute`.
    * `OpKernelKey` is the map key. Only device place now, but may be data type later.

---

# Why separate Kernel and Operator

* Separate GPU and CPU code.
    * Make Paddle can run without GPU.
* Make one operator (which is user interface) can contain many implementations.
    * Same mul op, different FP16, FP32 Kernel. different MKL, eigen kernel.
---

# Libraries for Kernel development

* `Eigen::Tensor` contains basic math and element-wise functions.
    * Note that `Eigen::Tensor` has broadcast implementation.
    * Limit number of `tensor.device(dev) = ` in your code.
* `thrust::tranform` and `std::transform`.
    * `thrust` has the same API as C++ standard library. Using `transform` can quickly implement a customized elementwise kernel.
    * `thrust` has more complex API, like `scan`, `reduce`, `reduce_by_key`.
* Hand-writing `GPUKernel` and `CPU` code
    * Do not write `.h`. CPU Kernel should be in `.cc`. GPU kernel should be in `.cu`. (`GCC` cannot compile GPU code.)
---
# Operator Register

## Why register is necessary?
We need a method to build mappings between Op type names and Op classes.

## How to do the register?

Maintain a map, whose key is the type name and value is corresponding Op constructor.

---
# The Registry Map

### `OpInfoMap`

`op_type(string)` -> `OpInfo`

`OpInfo`:

- **`creator`**: The Op constructor.
- **`grad_op_type`**: The type of the gradient Op.
- **`proto`**: The Op's Protobuf, including inputs, outputs and required attributes.
- **`checker`**: Used to check attributes.

---
# Related Concepts

### Op_Maker
It's constructor takes `proto` and `checker`. They are compeleted during Op_Maker's construction. ([ScaleOpMaker](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/scale_op.cc#L37))

### Register Macros
```cpp
REGISTER_OP(op_type, op_class, op_maker_class, grad_op_type, grad_op_class)
REGISTER_OP_WITHOUT_GRADIENT(op_type, op_class, op_maker_class)
```

### `USE` Macros
make sure the registration process is executed and linked.

---
# Register Process
1. Write Op class, as well as its gradient Op class if there is.
2. Write Op maker class. In the constructor, describe its inputs, outputs, and attributes.
3. Invoke macro `REGISTER_OP`. The macro will
	1. call maker class to complete `proto` and `checker`
	2. with the completed `proto` and `checker`, build a new key-value pair in the `OpInfoMap`

4. Invoke `USE` macro in where the Op is used to make sure it is linked.

---
# Backward Module (1/2)
### Create Backward Operator
- Mapping from forwarding Op to backward Op
![backward](https://gist.githubusercontent.com/dzhwinter/a6fbd4623ee76c459f7f94591fd1abf0/raw/61026ab6e518e66bde66a889bc42557a1fccff33/backward.png)

---
# Backward Module (2/2)
### Build Backward Network
- **Input** graph of forwarding operators
- **Output** graph of backward operators
- **corner case in construction**
	- shared variable => insert `Add` operator
	- no gradient => insert `fill_zero_grad` operator
	- recursive netOp => call `Backward` recursively
	- RNN Op => recursively call `Backward` on stepnet


---
# Scope, Variable, Tensor

* `Tensor` is an n-dimension array with type.
	* Only dims and data pointers are stored in `Tensor`.
	* All operators on `Tensor` is written in `Operator` or global functions.
	* variable length Tensor design [LoDTensor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/lod_tensor.md)
* `Variable` is the inputs and outputs of an operator. Not just `Tensor`.
	* step_scopes in RNN is a variable and not a tensor.
* `Scope` is where variables store at.
	* map<string/*var name */, Variable>
	* `Scope` has a hierarchical structure. The local scope can get variable from its parent scope.

---
# Block (in design)
## the difference with original RNNOp
- as an operator is more intuitive than `RNNOp`,
- offers new interface `Eval(targets)` to deduce the minimal block to `Run`,
- fits the compile-time/ runtime separation design.
  - during the compilation, `SymbolTable` stores `VarDesc`s and `OpDesc`s and serialize to a `BlockDesc`
  - when graph executes, a Block with `BlockDesc` passed in creates `Op` and `Var` then `Run`

---
# Milestone
- take Paddle/books as the main line, the requirement of the models motivates framework refactoring,
- model migration
  - framework development gives **priority support** to model migration, for example,
    - the MNIST demo needs a Python interface,
    - the RNN models require the framework to support `LoDTensor`.
  - determine some timelines,
  - heavily-relied Ops need to be migrated first,
  - different models can be migrated parallelly.
- improve the framework at the same time
- accept imperfection, concentrated on solving the specific problem at the right price.

---
# Control the migration quality
- compare the performance of migrated models with old ones.
- follow google C style
- build the automatic workflow of generating Python/C++ documentations
  - the documentation of layers and ops should be written inside the code
  - take the documentation quality into account when doing PR
  - preview the documentations, read and improve them from users' perspective
