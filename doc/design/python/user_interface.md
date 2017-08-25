# User Interface Design

## Basic Concepts
### Variable
A `Variable` represents shared, persistent state manipulated by a Paddle model program.

Variables are maintained by `pd.Variable` class,
each `pd.Variable` represents a tensor whose value can be changed by running ops on it.

A basic way to create a variable:

```python
import paddle as pd

v = pd.Variable(shape=[20, 20])
```

to get the value of the variable, one can call

```python
print v.val()
```

By default, Variables are model parameters, and will be updated after the network's back propagation.

One can freeze a variable by setting `trainable` to `False` like:

```python
v = pd.Variable(shape=[20,20], trainable=False)
```

### Block
Paddle use a `Block` to represent user's program, 
this is a basic concept when user write a Paddle program.

The function of `Block` is to enable groups of operators to be treated as if they were one operator, for example, when using a `RNNOp`, we can use block to help configure a step network:

```python
v = some_op()
m_boot = some_op()

W = pd.Variable(shape=[20, 20])
U = pd.Variable(shape=[20, 20])

rnn0 = RNNOp()
with rnn0.stepnet(inputs=[v]) as net:
    # declare stepnet's inputs
    x = net.add_input(v)
    # declare memories
    h = net.add_memory(m_boot)

    fc_out = pd.matmul(W, x)
    hidden_out = pd.matmul(U, h)
    sum = pd.add_two(fc_out, hidden_out)
    act = pd.sigmoid(sum)

    # declare stepnet's outputs
    net.add_output(act, hidden_out)

acts, hs = rnn0()
```

The operators inside the `with`-statement defines the rnn's step network, 
and will be put into a `pd.Block`.

another example is the definition of `if_else_op`:

```python
# v0 is a output of some_op
v0 = some_op()
v1 = some_op()

ifelseop = pd.if_else_op()
with ifelseop.true_block() as net:
    x0, x1 = net.add_input(v0, v1)
    
    y = pd.fc(x)
    z = pd.add_two(x1, y)
    
    net.add_output(y)

with ifelseop.false_block() as net:
    x0, x1 = net.add_input(v0, v1)
    
    y = pd.add_two(x0, x1)
    
    net.add_output(y)
    
# output of ifelseop
out = ifelseop()
```

### Op (short for Operator)
### Layer
### Special Ops
#### Initialize Operator
#### Optimizer Op

## Compatible with V2 Syntax

## Some Demos
### MNist Task Demo
### GAN Task Demo
