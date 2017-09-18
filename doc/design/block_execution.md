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


Since a block is actually a graph, the data member of a graph should be Nodes and Edges. Every Node must have a device id for descirbing the place information. Please refer to the design of [TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def.proto#L48). In our block design now, the data members are VarDesc and OpDesc. So, VarDesc and OpDesc must have a field to descirbe concrete device information. 

Let's consider about this question, What configuration should Paddle get from uses to execute a block in various hardward environment.

  
- Data Parallelism
  - stand-alone machine: Users have to set several device ids, and each device will execute the same block.
  - cluster: Users have to set node ids and device ids, and each device will execute the same block.
  
- Model Parallelism
  - Users have to set a concrete device id for each operator in a topology. Paddle will first execute the operator strictly following users' configuration, and will do some another parallel optimization automatically.

 


## Solution
  
Actually, we can have both data parallelism and model parallelism. In order to support the mixing paralelism, we have to propose these two strategies:


- A simple device id setting rules
  - Firsly, users have to set one or several default device id for a topology. These devices are the same and usually finish the most computation of a topology. We mainly consider data parallelism in this step.
  - Secondly, users need to set device id for each operators in a topology. If users set nothing, the operator will just use default device id. When devices have to be switched, users have to add a copy operator manually.
  - We travel the topology and set concrete device id for each operator.


Here is a pre-sudo code about the device setting rules.


```
import paddle as pd

def fpga_infer(fpga_id):
    with pd.device(fpga_id):
        x = pd.v2.variable(data_type='float32', dims=[32, 784], device='cpu')
        w = pd.v2.variable(data_type='float32', dims=[784, 100], device='cpu')
        x_fpga = pd.v2.copy_op(x)
        w_fpga = pd.v2.copy_op(w)
        h = pd.v2.mul_op(x_fpga, w_fgpa)
        h_gpu = pd.v2.copy(h, device='gpu:0')
        out = pd.v2.relu_op(h_gpu, device='gpu:0')
        out_cpu = pd.v2.copy_op(out, device='cpu')
    return out_cpu

pd.run(fpga_infer, devices=['fpga:0', 'fpga:1'])    
    
```

- A data dependency analysis engine
  - We need to analysis data dependency of each operator in the topology. When the dependent variables of a operator is ready, the operator can be executed in corresponding device.

  

The concept `Block` contains full information for executing a neural network topology. And we will have another concept `Executor` which have a runtime data depenency analysis engine to execute operators in `Block` in certain hardware with high performance.
  
  
## Conclusion

We will split our implementation of `Block` into following steps:

1. Baseline feature

2. Divide `Block` into two concepts, `Block` and `Executor`

3. Implement a simple device id setting rules

4. Data parallelism feature(run sequentially in each device)

5. Implement a data dependency analysis engine

6. Model parallelism feature
  



  
  
