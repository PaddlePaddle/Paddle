# Design Doc: Fully Static Graph

## Abstract

We propose the *fully static graph* rule: training and inference must
be fully specified by the static graph. This means training and
inference should be able to run solely on the cpp core (no Python
involved), everything should be implemented as an OP.

The user can still use Python to achieve the same result for
convenience when experimenting locally, but the distributed training
will not support Python.

## Background

There are two paradigms for expressing the computation graph: dynamic
and static. The dynamic paradigm constructs the graph on the fly:
every time `eval` is called, a new graph is created. The static
paradigm constructs the graph first, and then calls `eval`. There is
no new graph created each time `eval` is called.

The dynamic graph has the advantage of being flexible but is highly
dependent on the host language (most commonly Python). The static
graph is not as flexible, but more optimization can be done since the
graph is known before computing happens. PaddlePaddle is using the
static graph approach since we are focused on production deployment
and cluster training, efficiency is the key.

This design doc is trying to address an important question for the
static graph approach: should the training logic be fully specified by
the static graph?

For example, it's common to control the graph evaluation from Python:

```Python
for i in range(10000):
	paddle.eval(train_op)
```

In the above example: the training logic is not fully specified by the
graph: Python still take the control of the training logic.


## Fully Static Graph

The training logic should be fully specified by the graph (but we
still support controlling the graph evaluation from Python). Because
Python adds complication for distributed training:

- The distributed training engine needs to place the computation graph
  onto different nodes, and add communication OPs for data across node
  boundaries. They are very hard to do if the training logic is not
  fully specified by the graph.

- For fault recovery, every runtime state needs to be saved. But the
  state in Python code (such as training loop index and data reader
  position) could not be saved.

- Allowing executing arbitrary Python code on Paddle Cloud make
  training data safety very hard if not impossible to control.


### Benefits

- A clear separation between graph declaration (current using Python)
  and graph execution. It's easier for us to add a new language
  binding (or invent our own deep learning graph specification
  language).

- Local or distributed graph execution is easier to optimize.

- Much easier to ensure training data safety on Paddle Cloud.


### Example

To give a concrete example, for loop is essential for the training:
with every loop, a new mini-batch is fed into the training
system. Under the fully static graph rule, we **must** implement the for
loop as an OP:

```Python
# pseudo code, we need to discuss the for loop interface
i = pd.Variable(0)
optimizer = paddle.op.Adam()
# specify the input file as the argument, or
# leave blank and specify using config when running on Paddle Cloud
input = paddle.op.recordIO("/home/data/input.recordio")
q_x, q_y = input[0], input[1]
loss = pd.op.square(pd.op.sub(pd.op.add(pd.op.mul(x, w), b), y))

def cond(i):
    return i < 10000

with pd.for_loop(cond, [i]) as loop
    # Dequeue a new example each iteration.
    x = q_x.dequeue()
    y = q_y.dequeue()
    optimizer.minimize(loss)
    pd.add(i, 1)

# or paddle.save_target(loop, "job.bin") and
# submit the saved file to Paddle Cloud.
paddle.eval(loop)
```

The above code can run on both locally and on Paddle Cloud.

For user's convenience, he can use the Python for loop:
```Python
optimizer = paddle.op.Adam()
input = paddle.op.recordIO("/home/data/input.recordio")
q_x, q_y = input[0], input[1]
x = q_x.dequeue()
y = q_y.dequeue()
loss = pd.op.square(pd.op.sub(pd.op.add(pd.op.mul(x, w), b), y))
train_op = optimizer.minimize(loss)
for i in range(10000):
	paddle.eval(train_op)
```

The above code can only run locally.
