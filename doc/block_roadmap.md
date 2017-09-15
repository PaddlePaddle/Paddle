## Overview

We use `Block` to describe our neural network's topology. Actually, a neural network will be executed in various hardware environment. Generally, we will implement data parallelism and model parallelism to get high performance and scalability.

Here are some features we want to support:

- Baseline
  - one thread in a stand-alone machine: A neural network is executed sequentially in a single CPU or GPU.

- Data Parallelism
  - multi-threads in a stand-alone machine: Only one copy of parameters needs to be storaged in CPU memory.
  - multi-GPUs in a stand-alone machine: Each GPU card will have a full copy of parameters in its own GPU memory.
  - multi-threads/multi-GPUs in a cluster: Each node in the cluster supports multi-threads/multi-GPUs.

- Model Parallelism
  - Operators in a topology can locate in different devices(CPU/GPU/FPGA) in a stand-alone machine. Users have to set device id for every operator manually.
  - We should notice another situation where operators in a topology can be executed in different CPU threads or CUDA streams to achieve high performace, but they are actually in the same device. It's scheduled by Paddle automatically, and users don't need to set specific CPU thread id or CUDA stream id. 

- Long-term
  - Paddle schedules CPU/GPU/FPGA in a stand-alone machine automatically, makes full use of them to get best performance when executing a neural network.
  - Paddle schedules CPU/GPU/FPGA in a cluster automatically, and overlaps communication and computation well to get best performance.

  
  
## Analysis
  
Let's consider about this question, What configuration should Paddle get from uses to execute a topology in various hardward environment.
  
- Data Parallelism
  - stand-alone machine: Users have to set several device ids, and each device will execute the same topology.
  - cluster: Users have to set node ids and device ids, and each device will execute the same topology
  
- Model Parallelism
  - Users have to set a concrete device id for each operator in a topology. Paddle will first execute the operator strictly following users' configuration, and will do some another parallel optimization automatically.



## Solution
  
Actually, we can have both data parallelism and model parallelism. In order to support the mixing paralelism, we have to propose these two strategies:


- A simple device id setting rules
  - Firsly, users have to set one or several default device id for a topology. These devices are the same and usually finish the most computation of a topology. We mainly consider data parallelism in this step.
  - Secondly, users need to set device id for each operators in a topology. If users set nothing, the operator will just use default device id. When devices have to be switched, users have to add a copy operator manually.
  - We travel the topology and set concrete device id for each operator.


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
  



  
  
