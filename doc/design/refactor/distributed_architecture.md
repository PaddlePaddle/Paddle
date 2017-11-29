# Design Doc: Distributed Training Architecture

## Abstract

PaddlePaddle version 0.10.0 uses the "trainer-parameter server" architecture. We run multiple instances of trainers (where each trainer runs the same model) and parameter servers for distributed training. This architecture serves well, but has few limitations:

1. There is a need to write special code that handles tasks which should only be run on a single trainer. E.g., initializing the model, saving the model etc.

2. Model parallelism is hard: It would need all the if-else branches conditioned on the trainer ID to partition the model onto the trainers, and eventually manually writing out the inter-model-shard communication code to communicate between different trainers.

3. The user can not directly specify the parameter update rule: This would need to modify the parameter server code and compile a new binary. This makes things more complicated for researchers: A lot of extra effort is required to make this work. Besides, the training job submission program may not allow running arbitrary binaries.

This design doc discusses PaddlePaddle's new distributed training architecture that addresses the above mentioned limitations.

## Analysis

The assumption is that the user writes the trainer program in either Python or C++.

### Limitation 1

There are two basic functionalities in the trainer program:

1. The training logic such as loading / saving the model and printing out the logs.
2. The neural network definition such as the definition of the data layer, the fully connected layer, the cost function and the
  optimizer.

When we train using PaddlePaddle v0.10.0 in a distributed fashion, multiple instances of the same Python code are run on different nodes, hence both: the
training logic as well as the neural network computation logic, is replicated.

The tasks that only need to be run once belong to the training logic. Hence if we only replicate the neural network computation part, and do **not**
replicate the training logic, the limitation mentioned above can be avoided.

### Limitation 2

Model parallelism means that a single model is partitioned into different components and each node runs one of the component separately. This comes at the extra cost of managing the
inter-model-shard communication between nodes.

PaddlePaddle should ideally be able to modify the neural network computation and figure out the support for model parallelism automatically. However, the
computation is only specified in Python code which sits outside of PaddlePaddle, hence PaddlePaddle can not support the feature in this setup.

Similar to how a compiler uses an intermediate representation (IR) so that the programmer does not need to manually optimize their code for most of the cases, we can have an intermediate representation in PaddlePaddle as well. The compiler optimizes the IR as follows:

<img src="src/compiler.png"/>

PaddlePaddle can support model parallelism by converting the IR so that the user no longer needs to manually perform the computation and operations in the Python component:

<img src="src/paddle-compile.png"/>

The IR for PaddlePaddle after refactoring is called a `Block`, it specifies the computation dependency graph and the variables used in the computation.

### Limitation 3

The user can not directly specify the parameter update rule for the parameter server in the Python module, since the parameter server does not use the same computation definition as the trainer. Instead, the update rule is baked inside the parameter server. The user can not specify the update rule explicitly.

This could be fixed by making the parameter server run the same computation definition as the trainer (the user's Python module). For a detailed explanation, refer to this document -
[Design Doc: Operation Graph Based Parameter Server](./dist_train.md)

## Distributed Training Architecture

The revamped distributed training architecture can address the above discussed limitations. Below is the illustration of how it does so:

<img src="src/distributed_architecture.png"/>

The major components in the architecture are: *PaddlePaddle Python*, *PaddlePaddle converter* and *PaddlePaddle runtime*.

### PaddlePaddle Python

PaddlePaddle Python is the Python library that user's Python code invokes, to read the data. build the neural network topology, start training, etc.

```Python
paddle.init()
input = paddle.op.recordIO("/home/data/mnist.recordio") # file stored on the cluster
img, label = input[0], input[1]
hidden = paddle.layer.fc(input=img, size=200, act=paddle.activation.Tanh())
prediction = paddle.layer.fc(input=img, size=10, act=paddle.activation.Softmax())
cost = paddle.layer.classification_cost(input=prediction, label=label)
optimizer = paddle.optimizer.SGD(cost, learning_rate=0.01)
session = paddle.session.NewRemote(num_trainer=3, num_ps=2, GPU_per_trainer=1)
for i in range(1000):
	_, cost_val = session.eval(targets=[cost, optimizer])
	print cost_val
```

The above code is what a typical Python trainer code is, the neural network topology is built using the helper functions such as `paddle.layer.fc`. Training is done by calling `session.eval` iteratively.

#### session.eval

As shown in the graph, `session.eval` sends the IR and the evaluation inputs or targets to the PaddlePaddle cluster for evaluation.
The targets can be any variable in the computation graph. When the target is say, the `optimizer` variable, the neural network will be optimized once. When the target is the `cost` variable, `session.eval` returns the cost value. Based on what the target is, an appropriate action is taken.

The Python `session` is a wrapper of the C++ `Session` class. For more information about `Session`, refer to this document - [Design Doc: Session](./session.md).

### PaddlePaddle Converter

The PaddlePaddle converter automatically converts the IR in the request (IR and evaluation inputs/targets) from PaddlePaddle Python to partitioned IRs and dispatches the new IRs and evaluation inputs/targets to different PaddlePaddle runtimes. Below are the steps that are followed :

1. Add a `feed` OP that feeds the eval inputs, and a `fetch` OP that fetches the eval targets to the IR.

2. Extract a new computation (sub)graph with the `feed` and `fetch` OPs as the boundary. The runtime does not need to run the OP that is not dependent on the `fetch` OP.

3. Optimize the computation graph.

4. Place the OPs in the graph onto different devices on different PaddlePaddle runtime according to a placement algorithm and the device constraints specified by the user.

5. Partition the graph according to runtime boundaries and add `send` / `recv` OP pair on the runtime boundaries.

6. Dispatch the partitioned graph to different PaddlePaddle runtimes.

7. PaddlePaddle runtimes with the `fetch` OP reports evaluation results back to the converter, the converter reports the evaluation results back to the PaddlePaddle Python.

The output IRs will be cached to optimize the conversion latency.


#### Placement Algorithm

Our first implementation will only support "trainer-parameter server" placement: the parameters, initializers, and optimizers are all placed on the PaddlePaddle runtimes with the parameter server role. Everything else will be placed on the PaddlePaddle runtimes with the trainer role. This has the same functionality as the "trainer-parameter server" architecture of PaddlePaddle v0.10.0, but is more generic and flexible.

In the future, a more general placement algorithm should be implemented, which makes placements according to the input IR, and a model of device computation time and device communication time. Model parallelism requires the generic placement algorithm.


### PaddlePaddle Runtime

The PaddlePaddle runtime owns multiple devices (e.g., CPUs, GPUs) and runs the IR. The runtime does not need to do OP placement since it is already done by the converter.


### Local Training Architecture

The local training architecture will be the same as the distributed training architecture, the difference is that everything runs locally, and there is just one PaddlePaddle runtime:

<img src="src/local_architecture.png"/>


### Training Data

In PaddlePaddle v0.10.0, training data is typically read with a [data reader](../reader/README.md) from Python. This approach is no longer efficient when training in a distributed fashion since the Python process no longer runs on the same node with the trainer processes. The Python reader will need to read from the distributed filesystem (assuming it has the required access) and send to the trainers, doubling the network traffic.

When doing distributed training, the user can still use Python data reader: the training data are sent with `session.eval`. However this should be used for debugging purpose only. The users are encouraged to use the read data OPs.


## References:

[1] [TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)

[2] [TensorFlow: A System for Large-Scale Machine Learning](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)
