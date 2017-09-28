## Overview
PaddlePaddle represents a neural network as a `Block`. The word block and graph are interchangable in the desgin of PaddlePaddle. A block is like a pair of curly braces in programming languages like C++ and Java, it includes some variables and a sequence of instructions, or, operators. Usually we run a block using a CPU thread. But we want to accelerate the execution of some operators using devices like GPU and FPGA.

There are some challenges here to achieve high performance and scalability when executing a `Block`:

1. Execution Plan. It is often that a kind of device cannot accelerate all operators in a block. This requires an execution plan to use multiple devices. For example, some computational operators run on the GPU and cross-node data communication operators like SendOp and RecvOp on the CPU.

2. Parallel Execution.
It is often that a computer has more than one acceleration devices. For example, most servers and portable computers like PX2 have more than one CUDA GPUs. We want to make full use of them. Intuitively there are two approaches: (1) running operators in a block on multiple devices, and (2) duplicating the block and running them on multiple devices. The former is also known as model parallelism and the latter data parallelism.

3. Variable Places. Devices often have their onboard memories, and it is usually much more efficient for them to operator variables in the onboard memories. The execution plan should include the placement of variables in a block.


Here are the features we want to support:

1. Baseline
  - a single thread runs a block on a single device, e.g., a CPU or a GPU.

2. Data Parallelism
  - multi-threads in a stand-alone machine: Only one copy of parameters needs to be storaged in CPU memory.
  - multi-GPUs in a stand-alone machine: Each GPU card will have a full copy of parameters in its own GPU memory.
  - multi-threads/multi-GPUs in a cluster: Each node in the cluster supports multi-threads/multi-GPUs.

3. Model Parallelism
  - Operators in a block can locate in different devices(CPU/GPU/FPGA) in a stand-alone machine. Users have to set device id for every operator manually.
  - We should notice another situation where operators in a block can be executed in different CPU threads or CUDA streams to achieve high performace, but they are actually in the same device. It's scheduled by Paddle automatically, and users don't need to set specific CPU thread id or CUDA stream id. 

4. Long-term
  - Paddle schedules CPU/GPU/FPGA in a stand-alone machine automatically, makes full use of them to get best performance when executing a neural network.
  - Paddle schedules CPU/GPU/FPGA in a cluster automatically, and overlaps communication and computation well to get best performance.

  
  
## Analysis

The execution of a neural network topology in PaddlePaddle is divided into two parts, complie-time and run-time.

At complie-time, a ProgramDesc will be generated. At run-time, the ProgramDesc will be executed on specific hardwares.

We can refer to the design of [computation-graphs](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/refactorization.md#computation-graphs).

### Compile-time analysis

In order to realize features mentioned in overview, we have to do several transform pass to ProgramDesc:

- forward pass: generate OpDesc and VarDesc for forward operators

- backward pass: generate OpDesc and VarDesc for backward operators

- placement pass: generate OpDesc and VarDesc for send/recv/copy operators. Data Parallelism and Model Parallelism will be considered at this stage. We have a placement policy to set concrete device id for every operator. And send/recv/copy operators will be inserted at correct position. 

- optimize pass: generate OpDesc and VarDesc for optimize operators

Each pass will modify the global ProgramDesc.

### Run-time analysis

At run-time, the ProgramDesc generated at compile-time is actually executed on a specific hardware environment. `Executor` is introduced to execute a ProgramDesc. And `Executor` will have a `Run` interface.

In PaddlePaddle, we use DeviceContext to manage hardware resources. There are CPUDeviceContext and CUDADeviceContext for CPU and GPU respectively.A CUDADeviceContext actually is associated with a specific CUDA stream. And Operator will run on a specfic DeviceContext. 

However, users' configuration of hardware is simple, maybe only some GPU card ids will be given. In placement pass of compile-time, only source device id and destination device id will bet set.

So, `Executor` need to have a `DeviceContextManager` to initialize some DeviceContext at the very beginning. `DeviceContextManager` maneges all DeviceContexts in all devices.

The `Executor` is defined as follows:

```
class Executor {
public:
  Executor(ProgramDesc*);
  void Run();
  
private:
  DeviceContextManager_;
};
```


## Solution

### Compile-time solution
  
#### Remove NetOp

At current code base, we have a NetOp defined as follows:

```
class NetOp : OperatorBase {
public:
  void Run(const Scope&, const platform::DeviceContext&);
  std::vector<std::unique_ptr<OperatorBase>> ops_;
};
```

We can compose some basic operators into a big operator, such [FCOp](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/fc_op.cc). And the backward operators generated by [Backward](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/backward.h) is often a NetOp.

There are mainly two breaking point of NetOp unpon the overall design:

- Backward pass should be do at compile-time, not run-time.
- A NetOp will corresponds only one OpDesc in BlockDesc. We will have a big backward NetOp and run sequentially. It's hard to do potential memory-resuse/kernel-fusion optimization.

We should generate a group of OpDesc at compile-time instead of a NetOp at run-time.

#### Symbolic API

Please refer to the survey [doc](https://github.com/QiJune/Paddle/blob/e90ec7783a1abe7f7627f97559cc46488e41cc7e/doc/design/graph_survey.md) on Computation Graph. Users will write a neural network topology with Symbolic API. And the composition of operators should be done at compile-time in this level too.

#### Unified Pass Interface

An abstract class `Converter` is defined to provide a Unified Pass Interface.

```
class Converter {
public:
  virtual void ApplyPass(ProgramDesc*) = 0;
};
```

#### Placement Policy

Placement Policy is designed to set device for every operator. Currently, we only need a simple priority rule to implement the simplest version.



### Run-time solution
We will have several class derived from `Executor` to provide different execution strategy. And ProgramDesc will be transformed accordingly.

There are mainly two kinds of `Executor`:

- SimpleExecutor

SimpleExecutor will do little things to ProgramDesc, and just construct operators and variables in order. The operators will be executed sequentially.


- DAGExecutor

```
class DAGExecutor : public Executor {
public:
  void Transform();
private:
  GraphView* view_;
};
```

DAGExecutor will have a `GraphView` of ProgramDesc, which transforms linear list of Operators into graph data structure. We can do further optimization based on graph data structure more conveniently. And the graph will be executed in parallel in DAGExecutor.
