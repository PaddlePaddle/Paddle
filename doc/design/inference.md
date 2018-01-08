# Design Doc: Inferencer

In fluid, a nueral network is represented as a protobuf message [ProgramDesc](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/program.md), the python wrapper of which is `Program`.
Given a `ProgramDesc`, it can be run on any execution environment.
In fluid, we call the execution environment `Runtime`, which includes `Place`, `Scope` and `Executor`.

## Representation of the Inference Network

In python, an inference network is defined as:

```python
image = fluid.layers.data(name='x', shape=[784], dtype='float32')
predict = fluid.layers.fc(input=image,
                          size=10,
                          act='softmax')
```

After training for serval passes, the parameters can be saved use `fluid.io.save_inference_model`, which will save the binary proto string of the network at the same time.
```python
fluid.io.save_inference_model(
                "./inference_model/", ["x"], [predict],
                exe)
```

The saved model contains everything of the inference network, including all operators and variables. Thus, the `inference_program` should be initilized by the model file or a pre-loaded buffer.

Given a `inference_program`, it is easy to derive a `load_program` which is composed of `load_op` and is responsible for initializing all the parameter variables in `inference_program`. `load_program` will be executed once and `inference_program` will be executed as many times as you need.

To summerize, a inferencer should:
- be initialized from files or from buffers
- be composed of two ProgramDesc, namely the `inference_program` and  `load_program`

All the initialization is designed to be done in constructor.

## Support of Switching Runtime

In fluid, the execution environment is composed of three key concepts: `Place`, `Scope` and `Executor`.

There are two types of Place in current framework, `CPUPlace` for CPU and `CUDAPlace` for CUDA GPU. `Scope` is independent to `Place`. Given the place, you need to define a `Executor`, and run the `Executor` among the `Scope`.

In Inferencer, the `Runtime` is declared as follows:

```c++
class Runtime {
  platform::Place* place;
  framework::Scope* scope;
  framework::Executor* executor;
};
```

With the definition of `Runtime`, the `Inferencer` will has following features:
- **Switch runtime**. Different `Runtime` can have either different of the same type of `Place`, with different `Scope` and `Executor`. An `Inferencer` can run on different `Runtime` at the same time independently.
- **Share parameters among different networks**. Users can run different `Inferencer`, which means different network, on the same `Runtime`, parameters with the same name will be shared.
- **Share parameters among different threads**. Multi-threads can be launched to run an `Inferencer` in parallel on the same `Runtime`.

## Overview of the Inference API

A simple design, users can use the core data structure, `Tensor` and `LoDTensor`, to feed input data and fetch output data.
An `Inferencer` should enable the following members and public interfaces:
- Members:
  - the pointer of the `inference_program`
  - the pointer of the `load_program`
  - vectors of string to record the `feed_var_names` and `fetch_var_names`
  - the pointer of current `Runtime`
- Important interfaces:
  - constructor, to initialize the `inference_program` and `load_program`. Once initialized, they cannot be changed.
  - `Run`, to run the inference based on the current runtime. 
  - `SetRuntime`, to set the current runtime. When the runtime is set, the `load_program` will be run once to load parameters from files. 
- Utility interfaces:
  - `GetFeed/FetchVarNames`, to help users to debug.
  - `GetFeed/FetchVarShape`, to help users to verify the size of input and output data.

```c++
class Inferencer {
 public:
  // Initialize from file
  Inferencer(const std::string& filename);
  // Initialize from buffer
  Inferencer(const char* buffer, const size_t num_bytes);
  
  void SetRuntime(Runtime* runtime);
  
  void Run(const std::vector<framework::Tensor>& feeds,
           std::vector<framework::Tensor>& fetchs);
  
  // utility inferfaces
  std::vector<std::string>& GetFeedVarNames() const;
  std::vector<std::string>& GetFecthVarNames() const;
  std::vector<int64_t> GetFeedVarShape(const size_t index);
  std::vector<int64_t> GetFetchVarShape(const size_t index);
  
 private:
  framework::ProgramDesc* inference_program_;
  framework::ProgramDesc* load_program_;
  std::vector<std::string> feed_var_names_;
  std::vector<std::string> fetch_var_names_;
  
  Runtime* runtime;
};
```

### Issues

- Normally, all fetching variables' names should be written in the ProgramDesc and read from file. If users want to add some extra fetching variables for debug, or for some other use, they need to regenerate the file again. Do we need to allow user to append extra fetching variables?
- How to support multi-devices?
