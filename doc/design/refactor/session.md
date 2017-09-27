# Design Doc: Session

## Abstract

The *session* object encapsulates the environment in which the
computation graph is executed.

We will have *local* session and *remote* session, they offer the
same [interface](#interface). The local session encapsulates the local
runtime environment and the remote session encapsulates the cluster
runtime envrionment.

The local runtime envrionment contains:

1. computation devices (i.e., CPU, GPU) handles, and
1. the [scope](../scope.md) which holds all variables.

The remote runtime envrionment contains:

1. computation devices (i.e., CPU and GPU on node 0, 1) in a cluster,
   and
1. the distributed [scope](../scope.md) in a cluster which holds all
   variables.

The user can create a remote session on Paddle Cloud and evaluate the
computation graph with it. In this way, the user can control the
remote computation resource in a cluster from his local computer.


## Background

The current design has an implicit global session on which
`paddle.eval()` is executed. The pain point is:

Since the user is not able to explicitly switch between runtime
environments such as the scope and the device contexts, the user
cannot run a topology in two independent environments.

For example, in reinforcement learning, the user may want to have a
stale model for inference and a fresh model for training, and only
replace the stale model with the fresh model periodically.

Furthermore, we have no concept that encapsulates a remote environment
that executes a computation graph.

We need the session object to address above issues.


## Session

A session is an object that owns the runtime environment. All
computations are executed through `session.eval`.


### Interface

```
eval(
    targets,
    feed_dict=None,
)
```

Evaluates the target Operations or Variables in `targets`.

- *targets*: the evaluation targets. Can be a single Operation or
  Variable, or a list with the Operations or Variables as elements.

  The value returned by `eval()` has the same shape as the `target`
  argument.

  The computation graph is implicitly inferred from the targets.

- *feed_dict*: a dictionary that contains the tensors which overrides
  the edges of the computation graph.

```
close()
```

Closes the session. Calling this method releases the scope.


### Create a Local Session

```
session(
    gpu_ids=None
)
```

Creates a new session. One session owns one scope, so creating
multiple sessions will create different scopes.

- *gpu_ids*: a single `int` or a list of `int` of the GPU IDs to be
  used as the computation devices. If not specified, all avaiable GPUs
  will be used.


#### Example

```Python
a = paddle.constant(1.0)
b = paddle.constant(2.0)
c = a + b
sess = paddle.session(gpu_ids=[0,1])
sess.eval(c)
sess.close()
```

### Create a Remote Session

```
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

```
get_cloud_job(
    name
)
```

Gets a Paddle Cloud job.

```
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
