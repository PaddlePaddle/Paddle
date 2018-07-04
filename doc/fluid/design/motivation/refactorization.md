# Design Doc: Refactorization Overview

The goals of refactoring include:

1. Making it easy for external contributors to write new elementary computation operations.
1. Making the codebase clean and readable.
1. Designing a new computation representation -- a computation graph of operators and variables.
1. Implementing auto-scalability and auto fault recoverable distributed computing with the help of computation graphs.

## Computation Graphs

1. PaddlePaddle represents the computation, training and inference of Deep Learning models, by computation graphs.

  1. Please refer to [computation graphs](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/others/graph.md) for a concrete example.

1. Users write Python programs to describe the graphs and run them (locally or remotely).

1. A graph is composed of *variables* and *operators*.

1. The description of graphs must be serializable/deserializable, so that:

   1. It can be sent to the cloud for distributed execution, and
   1. It can be sent to clients for mobile or enterprise deployment.

1. The Python program does two things

   1. *Compilation* runs a Python program to generate a protobuf message representation of the graph and send it to
      1. the C++ library `libpaddle.so` for local execution,
      1. the master process of a distributed training job for training, or
      1. the server process of a Kubernetes serving job for distributed serving.
   1. *Execution* executes the graph by constructing instances of class [`Variable`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/variable.h#L24) and [`OperatorBase`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/operator.h#L70), according to the protobuf message.

## Description and Realization of Computation Graph

At compile time, the Python program generates a protobuf message representation of the graph, or a description of the graph.

At runtime, the C++ program realizes the graph and runs it.

<table>
<thead>
<tr>
<th></th>
<th>Representation (protobuf messages)</th>
<th>Realization (C++ class objects) </th>
</tr>
</thead>
<tbody>
<tr>
<td>Data</td>
<td>
<a href="https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/framework.proto#L107">VarDesc</a></td>
<td>
<a href="https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/variable.h#L24">Variable</a></td>
</tr>
<tr>
<td>Operation </td>
<td>
<a href="https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/framework.proto#L35">OpDesc</a></td>
<td>
<a href="https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/operator.h#L64">Operator</a></td>
</tr>
<tr>
<td>Block </td>
<td>BlockDesc </td>
<td>Block </td>

</tbody>
</table>


The word *graph* is interchangeable with *block* in this document.  A graph consists of computation steps and local variables similar to a C++/Java program block, or a pair of parentheses(`{` and `}`).

## Compilation and Execution

1. Run a Python program to describe the graph.  In particular, the Python application program does the following:

   1. Create `VarDesc` to represent local/intermediate variables,
   1. Create operators and set attributes,
   1. Validate attribute values,
   1. Infer the type and the shape of variables,
   1. Plan memory-reuse for variables,
   1. Generate the backward graph
   1. Add optimization operators to the computation graph.
   1. Optionally, split the graph for distributed training.

1. The invocation of `train` or [`infer`](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/inference.py#L108) methods in the Python program does the following:

   1. Create a new Scope instance in the [scope hierarchy](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/scope.md) for each run of a block,
      1. realize local variables defined in the BlockDesc message in the new scope,
      1. a scope is similar to the stack frame in programming languages,

   1. Create an instance of class `Block`, in which,
      1. realize operators in the BlockDesc message,

   1. Run the Block by calling
      1. `Block::Eval(vector<Variable>* targets)` for forward and backward computations, or
      1. `Block::Eval(vector<Operator>* targets)` for optimization.


## Intermediate Representation (IR)

```text
Compile Time -> IR -> Runtime
```

### Benefits of IR

- Optimization
  ```text
  Compile Time -> IR -> Optimized IR -> Runtime
  ```
- Automatically send partitioned IR to different nodes.
  - Automatic Data Parallelism
    ```text
    Compile Time
    |-> Single GPU IR
        |-> [trainer-IR-0, trainer-IR-1, pserver-IR]
            |-> Node-0 (runs trainer-IR-0)
            |-> Node-1 (runs trainer-IR-1)
            |-> Node-2 (runs pserver-IR)
    ```
  - Automatic Model Parallelism (planned for future)

---

## Operator/OpWithKernel/OpKernel

![class_diagram](https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/op_op_with_kern_class_diagram.dot)

---

## Operator
![class_diagram](https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/op.dot)

* `Operator` is the fundamental building block of the user interface.
    * Operator stores input/output variable names and attributes.
    * The `InferShape` interface is used to infer the shape of the output variables based on the shapes of the input variables.
    * Use `Run` to compute the `output` variables from the `input` variables.

---

## OpWithKernel/Kernel

![class_diagram](https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/op_with_kernel.dot)

* `OpWithKernel` inherits `Operator`.
* `OpWithKernel` contains a Kernel map.
    * `OpWithKernel::Run` get device's kernel, and invoke `OpKernel::Compute`.
    * `OpKernelKey` is the map key. Only device place now, but may be data type later.

---

## Why separate Kernel and Operator

* Separate GPU and CPU code.
    * Make Paddle capable of running without GPU.
* Make one operator (which is a user interface) and create many implementations.
    * For example, same multiplication op can have different implementations kernels such as FP16 kernel, FP32 kernel, MKL, eigen kernel.
---

## Libraries for Kernel development

* `Eigen::Tensor` contains basic math and element-wise functions.
    * Note that `Eigen::Tensor` has broadcast implementation.
    * Limit the number of `tensor.device(dev) = ` in your code.
* `thrust::transform` and `std::transform`.
    * `thrust` has the same API as C++ standard library. Using `transform`, one can quickly implement customized element-wise kernels.
    * `thrust`, in addition, supports more complex APIs, like `scan`, `reduce`, `reduce_by_key`.
* Hand-writing `GPUKernel` and `CPU` code
    * Do not write in header (`.h`) files. CPU Kernel should be in cpp source (`.cc`) and GPU kernels should be in cuda (`.cu`) files. (GCC cannot compile GPU code.)
---
## Operator Registration

### Why is registration necessary?
We need a method to build mappings between Op type names and Op classes.

### How is registration implemented?
Maintaining a map, whose key is the type name and the value is the corresponding Op constructor.

---
## The Registry Map

### `OpInfoMap`

`op_type(string)` -> `OpInfo`

`OpInfo`:

- **`creator`**: The Op constructor.
- **`grad_op_type`**: The type of the gradient Op.
- **`proto`**: The Op's Protobuf, including inputs, outputs and required attributes.
- **`checker`**: Used to check attributes.

---
## Related Concepts

### Op_Maker
It's constructor takes `proto` and `checker`. They are completed during Op_Maker's construction. ([ScaleOpMaker](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/scale_op.cc#L37))

### Register Macros
```cpp
REGISTER_OP(op_type, op_class, op_maker_class, grad_op_type, grad_op_class)
REGISTER_OP_WITHOUT_GRADIENT(op_type, op_class, op_maker_class)
```

---
## Registration Process
1. Write an Op class and its gradient Op class, if required.
2. Write an Op maker class. In the constructor of this class, describe the inputs, outputs and attributes of the operator.
3. Invoke the macro `REGISTER_OP`. This macro will
	1. Call maker class to complete `proto` and `checker`
	2. Using the completed `proto` and `checker`, it will add a new key-value pair to the `OpInfoMap`

---
## Backward Module (1/2)
### Create Backward Operator
- Mapping from forward Op to backward Op
![backward](https://gist.githubusercontent.com/dzhwinter/a6fbd4623ee76c459f7f94591fd1abf0/raw/61026ab6e518e66bde66a889bc42557a1fccff33/backward.png)

---
## Backward Module (2/2)
### Build Backward Network
- **Input**: a graph of forward operators
- **Output**: a graph of backward operators
- **Corner cases in construction**
	- Shared Variables => insert an `Add` operator to combine gradients
	- No Gradient => insert a `fill_zero_grad` operator
	- Recursive NetOp => call `Backward` recursively
	- RNN Op => recursively call `Backward` on stepnet
	- RNN Op => recursively call `Backward` on stepnet


---
## Scope, Variable, Tensor

* `Tensor` is an n-dimension array with type.
	* Only dims and data pointers are stored in `Tensor`.
	* All operations on `Tensor` are written in `Operator` or global functions.
	* Variable length Tensor design [LoDTensor](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/lod_tensor.md)
* `Variable` instances are the inputs and the outputs of an operator, not just `Tensor`.
	* `step_scopes` in RNN is a variable and not a tensor.
* `Scope` is where variables are stored.
	* map<string `var name`, Variable>
	* `Scope` has a hierarchical structure. The local scope can get variables from its parent scope.

---
## Block (in design)
### the difference between original RNNOp and Block
- As an operator is more intuitive than `RNNOp`,
- Offers a new interface `Eval(targets)` to deduce the minimal block to `Run`,
- Fits the compile-time/ runtime separation design paradigm.
  - During the compilation, `SymbolTable` stores `VarDesc`s and `OpDesc`s and serialize to a `BlockDesc`
  - When graph executes, a Block with `BlockDesc` is passed. It then creates `Op` and `Var` instances and then invokes `Run`.

---
## Milestone
- Take Paddle/books as the main line, the requirement of the models motivates framework refactoring,
- Model migration
  - Framework development gives **priority support** to model migration, for example,
    - the MNIST demo needs a Python interface,
    - the RNN models require the framework to support `LoDTensor`.
  - Determine some timelines,
  - Frequently used Ops need to be migrated first,
  - Different models can be migrated in parallel.
- Improve the framework at the same time
- Accept imperfection, concentrate on solving the specific problem at the right price.

---
## Control the migration quality
- Compare the performance of migrated models with old ones.
- Follow the google C++ style guide.
- Build the automatic workflow of generating Python/C++ documentations.
  - The documentation of layers and ops should be written inside the code.
  - Take the documentation quality into account when submitting pull requests.
  - Preview the documentations, read and improve them from a user's perspective.
