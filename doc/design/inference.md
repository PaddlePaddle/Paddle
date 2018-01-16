# Design Doc: Inference Engine

- [overview](#overview)
- [Inference Program](#inference-program)
  - [Introduction of Program Builder](#introduction-of-program-builder)
  - [Support for Common Training](#support-for-common-training)
- [Execution Runtime](#execution-runtime)
- [Program Resolver](#program-resolver)
  - [Design of Program Resolver](#design-of-program-resolver)
  - [Why do we need a Program Resolver?](#why-do-we-need-a-program-resolver)

The main goal of an inference API is to make it easy to use.
In Fluid, a neural network is represented as a protobuf message [ProgramDesc](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/program.md) called, the Python wrapper of which is a [Program](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/fluid/framework.py).
Given an [inference program](#inference-program), it can be executed inside any execution environment.
In Fluid, we call the execution environment a runtime, which includes a [Place](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/platform/place.h), a [Scope](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/scope.md) and an [Executor](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/executor.md).

## Overview

There are two global `Program`s defined in Python API of Fluid, namely `_main_program_` and `_startup_program_` respectively in [framework.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/fluid/framework.py).
They are referenced as `default_main_program` and `default_startup_program`, and usually used to construct a training program. Take an example, when defining a `fluid.layers.fc`, a `mul`, a `elementwise_add` and a activation operator are appended to the `default_main_program` to do the computation of `f(W * x + b)`, an `uniform_random` and a `fill_constant` operator are appended to the `default_startup_program` to initialize the paramaters `W` and `b`.

There are always a `main_program` and a `startup_program` in Fluid tasks. The `main_program` defines the computational operators and all variables, and can be evaluated as many times as the users want. The `startup_program` program is responsible for initializing all the persistable variables. It usually needs to be evaluated for a specified executor only once.

## Inference Program

There are three ways to define an inference program.
- **Case 1**, split from a training program. A training program can provide the inference serving at the same time, in which case the inference program is part of the training program, and all the parameters have been set correctly. There is no need of an extra `startup_program` for this kind of inferencing now and the need of an separate `main_program` for inference may be removed in the future which depends on the implementation of `Executor.Run()`.
- **Case 2**, write an inference program directly using API. In this case, parameters are stored in files.
- **Case 3**, read a pre-trained inference program from file. In this case, both the `ProgramDesc` and parameters are stored in files. We can get a complete `ProgramDesc` straightway and keeping a `main_program` and a `startup_program` make it possible to perform some online optimization (discussed [below](#introduction-of-program-builder)).

In this design doc, we mainly detail the interfaces for the **Case 3**.
- The protobuf message of the `main_program` is saved using `fluid.io.save_inference_model` method. Thus, it can be initilized from file or from a pre-loaded buffer.
- Since all the parameters are saved to files, the `startup_program` is initially composed of `load_op`s and  There is no need to save the protobuf message of the `startup_program` because it can be easily derived from the `main_program`.

A simple inference program can be defined in Python API as the:

```python
image = fluid.layers.data(name='x', shape=[784], dtype='float32')
predict = fluid.layers.fc(input=image,
                          size=10,
                          act='softmax')
```

After training for several epochs, the parameters can be saved using the method [fluid.io.save_inference_model](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/fluid/io.py), which will save the binary proto string of the program at the same time.

```python
fluid.io.save_inference_model(
                "./inference_model/", ["x"], [predict],
                exe)
```

### Introduction of Program Builder

The Python implementation of `Program` is not exactly equal to the C++ implementation of `ProgramDesc`. It provides some additional functions, such as `inference_optimize` and `append_backward`. It also records the index of current block in program.
We introduce a similar concept of a `ProgramBuilder` in C++, which collects all the metadata of an program. Specially, it supports transformation and optimization for an inference program.

```cpp
class ProgramBuilder {
 public:
  // Initialize an empty program
  ProgramBuilder();
  // Initialize from file
  ProgramBuilder(const std::string& filename);
  // Initialize from buffer
  ProgramBuilder(const char* buffer, const size_t num_bytes);

  framework::ProgramDesc* MainProgram();
  framework::ProgramDesc* StartupProgram();

  // Some utility interface maybe required by users
  std::vector<std::string>& FeedVarNames() const;
  std::vector<std::string>& FetchVarNames() const;
  std::vector<int64_t> FeedVarShape(const size_t index);
  std::vector<int64_t> FetchVarShape(const size_t index);

  void AppendFetchVariables(const std::string& var_name);
  ...

  // Perform transformation and optimization of the inference program
  ProgramBuilder* operator()(/* some optimizing strategy */);
  ProgramBuilder* operator()(const std::vector<std::string>& feed_var_names,
                             const std::vector<std::string>& fetch_var_names,
                             /* some optimizing strategy */);
  ProgramBuilder* operator()(const std::vector<framework::VarDesc>& targets,
                             /* some optimizing strategy */);

  // Support for training
  ProgramBuilder* Clone();
  void AppendBackward(std::vector<framework::Variable>& targets, std::vector<framework::Variable>& no_grad_set);

 private:
  framework::ProgramDesc* main_program_;
  framework::ProgramDesc* startup_program_;
  std::vector<std::string> feed_var_names_;
  std::vector<std::string> fetch_var_names_;
};
```

In the first design, `ProgramBuilder` contains all the elements mentioned above, and is instantiated by protobuf message of the `main_program`. Other members `startup_program`, `feed_var_names` and `fetch_var_names` will also be derived in the constructor.

There are two advantages of introducing an independent concept of a `ProgramBuilder`:
- It is easy to add utility interfaces to support other requirements.
  For example,
  - `Feed/FetchVarNames`. It can be used to help users verify how many inputs and outputs are reqiured and what the names of those are.
  - `Feed/FetchVarShape`. It can be used to help users verify the size of each input and output.
  - `AppendFetchVariables`. Normally, the names of all the variables to be fetched should be included in the protobuf message of the `main_program`. However, sometimes users may want to fetch extra variables for other use or debugging purposes, they can use this interface directly and there would be no need to regenerate the protobuf message again. Note that `main_program` may be modified in this interface.
- It is possible to support online optimization of the inference program.
  We will design an inference transpiler to do offline optimization for inference, which will result into an optimized inference `ProgramDesc` for a given `ProgramDesc`. However, some optimization can be done online, for example:
  - changing the layout from `NCHW` to `NHWC`
  - merging the computation of batch normalization layer to the front fc layer or conv layer

  `ProgramBuilder` overrides the `()` operator to support this kind of optimization, in which both `main_program` and `startup_program` may be modified. Thus, users may specify some optimizing stategy and will get a new instance of `ProgramBuilder`.

### Support for Common Training

Based the concept of `ProgramBuilder`, it is easy to implement a common C++ API to support training.

- Define default main program and startup program for training.

```c++
ProgramBuilder* default_builder = std::unique_ptr<ProgramBuilder>(new ProgramBuilder()).get();
ProgramDesc* default_main_program() { return default_builder->MainProgram(); }
ProgramDesc* default_startup_program() { return default_builder->StartupProgram(); }
ProgramBuilder* switch_main_program(ProgramBuilder* new_builder) {
  ProgramBuilder* prev_builder = default_builder;
  default_builder = new_builder;
  return prev_builder;
}
```

- Implement C++ wrapper for each layer.

```c++
namespace fluid {
namespace layers {
framework::VarDesc& data(std::string& name, std::vector<int64_t>& shape, framework::proto::DataType type) {
 ...
}

framework::VarDesc& fc(framework::VarDesc& input, size_t size, int num_flatten_dims = 1, std::string act, ...) {
  framework::OpDesc* op = default_main_program().CurrentBlock().AppendOp();
  op->SetType("mul");
  ...
}
}  // namespace layers
}  // namespace fluid
```

Then, users can write a training program using following codes.

```c++
auto image = fluid::layers::data("x", {784}, framework::proto::DataType::FP32);
auto hidden = fluid::layers::fc(image, 10, 1, "softmax");
...
```

## Execution Runtime

There are three key concepts in Fluid: `Place`, `Scope` and `Executor`.
- `Place` is used to decide which device the program will run on. There are two types of `Place` in the current framework, `CPUPlace` for CPU and `CUDAPlace` for CUDA GPU.
- `Scope` in Fluid is similar to the concept of a `Scope` in programming languages. It is an association of a name to variable. Global variables in the same `Scope` should have different names. However, there is no restrictions on names of variables in different local scopes. Users have to specify a `Scope` to run a program.
- `Executor` can be constructed by a user specified place, and provides a unified way to execute a `ProgramDesc` in a `Scope`.

All three concepts compose the execution environment, that is a `Runtime` for inference.

```c++
class Runtime {
 public:
  Runtime(/* CPU or GPU */);

 private:
  platform::Place* place;
  framework::Scope* scope;
  framework::Executor* executor;
};
```

1. A program can run on different `Runtime`s.
   Users can define a runtime for CPU and another runtime for CUDA GPU, and the inference program can run on the two runtimes at the same time. Or users can define two runtimes for CUDA GPU to run the inference program on different GPU devices.
1. It is possible to share parameters amongst different programs.
   Different programs can run on the same `Runtime`, so that parameters with the same name will be shared.
1. Programs running on different threads can share parameters.
   Multi-threads can be launched to run an inference program in parallel on the same `Runtime`.

## Program Resolver

### Design of Program Resolver

Similar to `Program`, the Python implementation of `Executor` is a simple wrapper of the C++ implementation of `Executor` as well. It hiddens much details for users, such as inserting `feed_op` and `fetch_op`, setting feed variables and getting fetch variables.
An similar concept `ProgramResolver` is introduced in C++ to simplify the usage and provide the possibility to support more features in the future.
1. An `ProgramResolver` doesn't own any computing resources and programs, but only holds a pointer to the current `Runtime`. Users can call `SetRuntime()` to set the current runtime.
1. After setting the current runtime, users can call `Run()` to run the `startup_program` once to initialize parameters, then run the `main_program` as many times as they require.
1. Data structure, [framework::Tensor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/tensor.md) and [framework::LoDTensor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/lod_tensor.md), are used in user implementation to feed input data and fetch output data.

```c++
class ProgramResolver {
 public:
  ProgramResolver();

  void SetRuntime(const Runtime* runtime);

  void Run(framework::ProgramDesc* program,
           const std::vector<framework::Tensor>& feeds,
           std::vector<framework::Tensor>& fetchs);

  void Run(framework::ProgramDesc* program,
           DataFeeder& feeder, ...);

 private:
  Runtime* runtime;
};
```

### Why do we need a Program Resolver?

- Hidden much detail of the framework and simplify the usage.

  Using a `ProgramBuilder` and a `Runtime`, users can write code to perform inference.
Apart from the concepts introduced in this design doc, users need to handle the details of feed and fetch data, by calling `framework::SetFeedVariable` and `framework::GetFetchVariable`.
A simple example is listed as follows. For training, users need to insert `feed_op` and `fetch_op` manually.

  Here is the simplest example to use `ProgramSolver` to build an inference program directly from file and run on a single CPU.

```cpp
Runtime runtime("CPU");

ProgramResolver resolver;
resolver.SetRuntime(&runtime);

ProgramBuilder inference_builder("mnist.paddle");

// Run the startup_program to initialize parameters for the runtime
resolver.Run(inference_builder.StartupProgram(), {}, {})

// Run the main_program multiple times
for (...) {
  framework::Tensor input;
  framework::Tensor output;
  resolver.Run(inference_builder.MainProgram(), {input}, {output});
}
```

- Support for online training and inference.

  All the concepts introduced here can be easily extended to support online training and inference. The difficulty to train in pure C++ side is how to feed data. A `DataFeeder` need to be carefully designed.

```cpp
struct DataFeeder {
  std::vector<Tensor> operator()(...);
};
```

  To support online training, because the composition of the program is changing, an asynchronous thread need to be launched in the `Run()` to parse, create and run operators at real-time. Another asynchronous thread need to be lauched to run a reader to feed data.
