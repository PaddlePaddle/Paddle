# Design Doc: Session

## Abstract

This design doc proposes to have an object called *Session* which
encapsulates the environment in which the computation graph is
executed.

## Background

A computation graph is executed in an environment which contains the
[scope](./scope.md) and other states. PaddlePaddle used to only have
an implicit global session on which `paddle.eval()` is executed.

This has the limitation that the user can not create two independent
environments. For example, in reinforcement learning, the user may
want to have a stale model for inference and a fresh model for
training, and only replace the stale model with the fresh model
periodically. Also, we have no concept that can encapsulate a remote
environment that could execute a computation graph.

## Session

Session is an object that owns all runtime states such as scope,
reader OP's file handles, connection to a remote PaddlePaddle cluster,
etc.

Session has two methods: `eval` and `close`. `eval` executes the
target OP in a given graph, and `close` closes the session and
releases all related resources:

```Python
a = paddle.constant(1.0)
b = paddle.constant(2.0)
c = a + b
sess = paddle.session()
sess.eval(c)
sess.close()
```

### Remote Session

Paddle Cloud will support user creating a remote session pointing to
the Paddle Cloud cluster. The user can send the computation graph to
be executed on the Paddle Cloud. In this way, the user can control a
cluster from her local computer:

```Python
reader = paddle.reader.recordio("/pfs/home/peter/mnist-train-*") # data stored on Paddle Cloud
image = reader.column(0)
label = reader.column(1)
fc1 = paddle.op.fc(image, size=256, act="sigmoid")
fc2 = paddle.op.fc(fc1, size=10, act="softmax")
cost = paddle.op.cross_entropy(fc2)
opt = paddle.optimizer.sgd(cost)

remote_config = ... # remote configuration such as endpoint, number of nodes and authentication.
sess = paddle.remoteSession(remote_config)
for i in range(1000):
    sess.eval(opt)
sess.close()
```
