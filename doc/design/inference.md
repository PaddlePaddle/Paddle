# Design Doc: InferenceEngine

The main goal of inference API is easy to use.
In Fluid, a neural network is represented as a protobuf message [ProgramDesc](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/program.md), the Python wrapper of which is [Program](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/fluid/framework.py).
Given a [inference program](#inference-program), it can run inside any execution environment.
In Fluid, we call the execution environment runtime, which includes [Place](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/platform/place.h), [Scope](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/scope.md) and [Executor](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/executor.md).

## Inference Program

A simple inference program may be defined in Python API as:

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

Like training, there is a `main_program` and a `startup_program` for inference.
- The `main_program` defines the computational operators and all variables and can be evaluated as many times as users want. The protobuf message of it is saved by `fluid.io.save_inference_model`. Thus, it can be initilized from file or a pre-loaded buffer.
- The `startup_program` program is responsible for initializing all parameters. Since all the parameters are saved to files, the `startup_program` is composed of `load_op`s and needs to be evaluated for a specified executor only one time. There is no need to save the protobuf message of the `startup_program` because it can be easily derived from the `main_program`. The `startup_program`.

### Introduce of ProgramBuilder

We introduce the concept of `ProgramBuilder`, which will collect all the metadata of inference program and support transform and optimization of the inference program.

```cpp
class ProgramBuilder {
 public:
  // Initialize from file
  ProgramBuilder(const std::string& filename);
  // Initialize from buffer
  ProgramBuilder(const char* buffer, const size_t num_bytes);

  // Some utility interface maybe required by users
  std::vector<std::string>& GetFeedVarNames() const;
  std::vector<std::string>& GetFecthVarNames() const;
  std::vector<int64_t> GetFeedVarShape(const size_t index);
  std::vector<int64_t> GetFetchVarShape(const size_t index);

  void AppendFetchVariables(const std::string& var_name);
  ...

  // Do transform to the inference program
  ProgramBuilder* operator()(/* some optimizing strategy */);
  ProgramBuilder* operator()(const std::vector<std::string>& feed_var_names,
                             const std::vector<std::string>& fetch_var_names,
                             /* some optimizing strategy */);

 private:
  framework::ProgramDesc* main_program_;
  framework::ProgramDesc* startup_program_;
  std::vector<std::string> feed_var_names_;
  std::vector<std::string> fetch_var_names_;
};
```

In the first design, `ProgramBuilder` contains all the elements memtioned above, and is instanced by protobuf message of the `main_program`. Other members `startup_program`, `feed_var_names` and `fetch_var_names` will also be derived in the constructor.

There are two advantages of introducing an independent concept `ProgramBuilder`:
- It is easy to add utility interfaces to support other requirements.
  For example,
  - `GetFeed/FetchVarNames`. It can be used to help users verify how many inputs and outputs there need and what the names are.
  - `GetFeed/FetchVarShape`. It can be used to help users verify the size of each input and output.
  - `AppendFetchVariables`. Normally, all fetching variables' names should be included in the protobuf message of the `main_program`. However, sometimes users may want to fetch extra variables for other use or debugging purposes, they can use this interface directly and have no need to regenerate the protobuf message again. Note that `main_program` may be modified in this interface.
- It is possible to support online optimization of the inference program.
  We will design an inference transpiler to do offline optimization for inference, which produce an optimized inference `ProgramDesc` for a given `ProgramDesc`. However, some optimization can be done online, such as
  - changing the layout from `NCHW` to `NHWC`
  - merging the computation of batch normalization layer to the front fc layer or conv layer

  `ProgramBuilder` overrides the `()` operator to support this kind of optimization, in which both `main_program` and `startup_program` may be modified. Thus, users may specify some optimizing stategy and will get a new instance of `ProgramBuilder`.

## Execution Runtime

There are three key concepts in Fluid: `Place`, `Scope` and `Executor`.
- `Place` is used to decide which device the program will run on. There are two types of `Place` in the current framework, `CPUPlace` for CPU and `CUDAPlace` for CUDA GPU.
- `Scope` in Fluid likes the concept in programming languages. It is an association of a name to variable. Global variables in the same `Scope` should have different names. However, there is no restrictions on names of variables in different local scopes. Users have to specify a `Scope` to run a program.
- `Executor` can be constructed by a user specified place, and provides a unified way to execute a `ProgramDesc` in a `Scope`.

All the three concepts compose the execution environment, that is `Runtime` for inference.

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
1. It is possible to share parameters among different programs.
   Different program can run on the same `Runtime`, so that parameters with the same name will be shared.
1. Program running on different threads can share parameters.
   Multi-threads can be launched to run an inference program in parallel on the same `Runtime`.

## Inference Engine

### Why need an Inference Engine?

With `ProgramBuilder` and `Runtime`, user can write codes to do inference.
Apart from the concepts introduced specially for inference in this design doc, users need handle the details of feed and fetch data, by calling `framework::SetFeedVariable` and `framework::GetFetchVariable`.
In addition, users need to run the `startup_program` manually to load parameters for each runtime.
A simple example is listed as following.

```cpp
ProgramBuilder builder("mnist.paddle");
Runtime runtime("CPU");

// Run the startup_program once to load parameters for the specified runtime
runtime.Executor()->Run(builder.StartupProgram(), runtime.Scope(), 0, true, true);

// Run the main_program many times
for (...) {
  framework::LoDTensor input;
  framework::SetFeedVariable(runtime.Scope(), input, ...);
  runtime.Executor()->Run(builder.MainProgram(), runtime.Scope(), 0, true, true);
  framework::LoDTensor output;
  framework::GetFetchVariable(runtime.Scope(), output, ...);
}
```

To simplify the interfaces, we design a new structure, `InferenceEngine`.

### Design of Inference Engine

1. An `InferenceEngine` can be constructed by a `ProgramBuilder`.
1. An `InferenceEngine` also holds pointer to the current `Runtime`. Users can call `SetRuntime()` to set the current runtime, and the `startup_program` will be run once to initialize parameters for this runtime.
1. After setting the current runtime, users can call `Run()` to run the inference program as many times as they required.
1. Data structure, [framework::Tensor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/tensor.md) and [framework::LoDTensor](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/lod_tensor.md), are used in user codes to feed input data and fetch output data.

```c++
class InferenceEngine {
 public:
  InferenceEngine(const ProgramBuilder& builder);

  void SetRuntime(Runtime* runtime);

  void Run(const std::vector<framework::Tensor>& feeds,
           std::vector<framework::Tensor>& fetchs);

 private:
  ProgramBuilder builder;
  Runtime* runtime;
};
```

### Example

Here is the simplest example to use `InferenceEngine` to build a inference program directly from file and run on a single CPU.

```cpp
ProgramBuilder builder("mnist.paddle");
Runtime runtime("CPU");

InferenceEngine engine(builder);
// Set the runtime, in which the startup_program will be ran to initialize parameters for the runtime
engine.SetRuntime(&runtime);

// Run the main_program many times
for (...) {
  framework::LoDTensor input;
  framework::LoDTensor output;
  engine.Run({input}, {output});
}
```
