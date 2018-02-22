# Design Doc: Session

## Abstract

The *session* object encapsulates the environment in which the
computation graph is executed.

We will have the *local* session and *remote* session, they offer the
same [interface](#interface). The local session encapsulates the local
runtime environment and the remote session encapsulates the cluster
runtime environment.

The local runtime environment contains:

1. computation devices (i.e., CPU, GPU) handles, and
1. the [scope](../scope.md) which holds all variables.

The remote runtime environment contains:

1. computation devices (i.e., CPU and GPU on node 0, 1) in a cluster,
   and
1. the distributed [scope](../scope.md) in a cluster which holds all
   variables.

The user can create a remote session on Paddle Cloud and evaluate the
computation graph with it. In this way, the user can control the
remote computation resource in a cluster from his local computer.


## Background

The current design has an implicit global session in which
`paddle.eval()` is executed. The pain point is:

Since the user is not able to explicitly switch between runtime
environments, the user cannot run a topology in two independent
environments.

For example, in reinforcement learning, the user may want to have a
stale model for inference and a fresh model for training, and only
replace the stale model with the fresh model periodically.

Furthermore, we have no concept that encapsulates a remote environment
that executes a computation graph.

We need the session object to address above issues.


## Session

A session is an object that owns the runtime environment. All
computations are executed through `session.eval()`.


### Interface

```python
eval(
    targets,
    feed_dict=None,
)
```

Evaluates the target Operations or Variables in `targets`.

- *targets*: the evaluation targets. Can be a single Operation or
  Variable, or a list with the Operations or Variables as
  elements. The value returned by `eval()` has the same shape as the
  `target` argument.

  The PaddlePaddle program is represented by
  the [ProgramDesc](../design/program.md), `eval()` will infer the
  ProgramDesc from the given targets and run the PaddlePaddle
  program. Please
  see
  [this graph](./distributed_architecture.md#local-training-architecture) for
  the detailed illustration for the local session
  and
  [this graph](./distributed_architecture.md#distributed-training-architecture) for
  the detailed illustration for the remote session.

- *feed_dict*: a dictionary that contains the tensors which override
  the edges of the computation graph.

  feed_dict not only can provide the input data, it can override any
  OP's input as well:

  ```python
  a = pd.constant(2.0, name="a")
  b = pd.variable(name="b")
  c = pd.mul(a,b)
  sess.eval(targets=c, feed_dict={"b":3.0}) # returns 6.0
  ```

```python
close()
```

Closes the session and releases the scope that the session owns.


### Create a Local Session

```python
session(
    devices=None
)
```

Creates a new session. One session owns one global scope, so creating
multiple sessions will create different scopes.

- *devices*: a single `string` or a list of `string` of device names,
  the corresponding devices will be the computation devices for
  `eval()`. If not specified, all available devices (e.g., all GPUs)
  will be used. The user doesn't need to specify the CPU device since
  it will be always used. Multiple sessions can use the same device.


#### Example

```Python
a = paddle.constant(1.0)
b = paddle.constant(2.0)
c = a + b
sess = paddle.session(devices=["gpu:0", "gpu:1", "fpga:0"])
sess.eval(c)
sess.close()
```

### Create a Remote Session

```python
create_cloud_job(
    name,
    num_trainer,
    mem_per_trainer,
    gpu_per_trainer,
    cpu_per_trainer,
    num_ps,
    mem_per_ps,
    cpu_per_ps,
)
```

Creates a Paddle Cloud job. Fails if the job name exists.

```python
get_cloud_job(
    name
)
```

Gets a Paddle Cloud job.

```python
remote_session(
    job
)
```

- *job*: the Paddle Cloud job.

#### Example

```Python
reader = paddle.reader.recordio("/pfs/home/peter/mnist-train-*") # data stored on Paddle Cloud
image = reader.column(0)
label = reader.column(1)
fc1 = paddle.op.fc(image, size=256, act="sigmoid")
fc2 = paddle.op.fc(fc1, size=10, act="softmax")
cost = paddle.op.cross_entropy(fc2, label)
opt = paddle.optimizer.sgd(cost)

job = paddle.create_cloud_job("test", 3, "1G", 1, 1, 2, "1G", 1)
sess = paddle.remote_ession(job)
for i in range(1000):
    sess.eval(opt)
sess.close()
```
