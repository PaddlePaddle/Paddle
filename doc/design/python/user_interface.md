# User Interface Design

## Basic Concepts
### Variable
A `Variable` represents shared, persistent state manipulated by a Paddle model program.

Variables are maintained by `pd.Variable` class,
each `pd.Variable` represents a tensor whose value can be changed by running ops on it.

A basic way to create a variable is:

```python
import paddle as pd

v = pd.Variable(shape=[20, 20])
```

To make it more converient to share a variable, each `pd.Variable` has a name, 
one can use a name to get or create a `pd.Variable` by calling `pd.get_variable`, for example:

```python
# same as 
v = pd.get_variable(name="v", shape=[20, 20])
```

By default, Variables are model parameters, and will be updated after the network's back propagation.

One can freeze a variable by setting `trainable` to `False` like:

```python
v = pd.Variable(shape=[20,20], trainable=False)
```

Some initizlization strategies may be applied to variables, for example, we may set a variable to zero or gaussian random.

```
v = pd.Variable(shape=[20,20], initializer=pd.zero_initializer())
z = pd.Variable(shape=[20,20], initializer=pd.gaussian_initializer(mean=0., std=0.1))
```

to get the value of the variable, one can call

```python
print v.val()
```


### Block
Paddle use a `Block` to represent and execute user's program, 
this is a basic concept when user write a Paddle program.

In computer programming, a block is a lexical structure of source code which is grouped together. 
In most programming languages, block is useful when define a function or some conditional statements such as `if-else`, `while`.

Similarlly, the function of `pd.Block` in Paddle is to enable groups of operators to be treated as if they were one operator to make `if_else_op` or RNNOp's declaration simpler and Python's `with` statement is used to make the codes look much like a block.

For example, when defining a `RNNOp`, we can use `pd.Block` to help configure a step network:

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
    
    net.add_output(z)

with ifelseop.false_block() as net:
    x0, x1 = net.add_input(v0, v1)
    
    y = pd.add_two(x0, x1)
    
    net.add_output(y)
    
# output of ifelseop
out = ifelseop()
```

In most cases, user need not to create a `pd.Block` directly, but it is the basis of a Paddle program:

- user's program is stored in `pd.Block`
- when we want to run the codes, we just need to execute a corresponding `pd.Block`

A `pd.Block` can has its own namespace, which makes it possible to hide the local variables from block block.

```python
W = pd.Variable(shape=[20, 20])

# a and b are outputs of some_op
a = some_op()
b = some_op()

with pd.Block('namespace0'):
    # W is a local variable and has its own value
    W = pd.Variable(shape=[20, 20])
    x = pd.matmul(W, a)
    y = x + b
    
with pd.Block('namespace1'):
    # W is the global variable
    z = pd.matmul(W, a)
    
# g use local variables in both namespace0 and namespace1
g = pd.add_two(y, z)
```

### Op (short for Operator)
`Op` defines basic operation unit of optimized computation graph in Paddle, one `Op` has several input and output variables, and some attributes.

Take `pd.matmul` for example, one can use it like this

```python
out = pd.matmul(a, b)
```
which means that a operator `pd.matmul` takes two variables `a` and `b` for input, 
and return a variable `out`.

### Layer
`Layer` defines a more complex operation which may combines several `Op`s, its usage is the same with `Op`.

Take `pd.fc` for example, one can use it like this
```python
out = pd.fc(in, param_names=['W'])
```
which means that the `pd.fc` takes an variable `in`, and set its `param_names` attribute to `['W']`, 
 which will determine the names of its parameters.

Both `Op` and `Layer` will be appended to current `pd.Block` when they are created,
and there will be a sequene of Ops/Layers in the `pd.Block`,
if the `pd.Block` is executed, all the Ops/Layers in this `pd.Block` will be called in order.

### Special Ops
#### Initializer Ops
These ops will initialize variables, for example, we may have 

- `pd.zero_initializer()`
- `pd.gaussian_random_initializer(mean, std)`

Each trainable variable has a initialize Op. 

#### Optimizer Ops
These ops will help to optimize trainable variables after backward propagation finished, 
each variable will have a optimizer.

## Compatible with V2 Syntax

## Some Demos
### MNist Task Demo

```python
import paddle as pd

# the first shape is None, which means the batch size of variable is not known.
image = pd.Variable(shape=[None, 128])
label = pd.Variable(shape=[None, 1])

# network config
W1 = pd.Variable('W1', shape=[128, 64])

fc_out = pd.matmul(image, W1)
prediction = pd.softmax(fc_out, size=10)

cost = pd.cross_entropy(prediction, label)

optimizer = pd.SGDOptimizer().minimize(cost)


# training details
def data_provider(path):
    images = []
    labels = []
    with open(path) as f:
        for no, line in enumerate(f):
            fs = line.split('\t')
            assert len(fs) == 2
            image_record = map(int, fs[0].split())
            label_record = [int(fs[1])]
            images.append(image_record)
            labels.append(label_record)
            if no > 0 and no % 100 == 0:
                yield np.array(images), np.array(labels)
            images = []
            labels = []


for pass_no in range(100):
    for batch_no, batch in enumerate(data_provider('./data.txt')):
        # train mode
        _, cost_ = pd.eval(
            [optimizer, cost], feeds={image: batch[0],
                                      label: batch[1]})
        print '%dth pass train cost: %f' % (pass_no, cost_)
        # test mode
        if batch_no > 0 and batch_no % 10 == 0:
            cost_ = pd.eval(cost)
            print '%dth pass test cost' % (pass_no, cost_)
```

### GAN Task Demo

```python
import paddle as pd

# input variable whose batch size is unknown now
X = pd.Variable(shape=[None, 128])

# Discriminator Net
# define parameters

# Generator Net
Z = pd.data(pd.float_vector(100))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    return np.random.uniform(-1, 1., size=[m, n])


def discriminator(x):
    # use block with namespace to hide local variables
    with pd.Block('discriminator') as block:
        # declare model parameters
        W1 = pd.get_variable(
            'W1',
            shape=[784, 128],
            initializer=pd.gaussian_random_initializer(std=0.1))
        b1 = pd.get_variable(
            'b1', data=np.zeros(128)
        )  # variable also support initialization using a  numpy data
        W2 = pd.get_variable('W2', data=np.random.rand(128, 1))
        b2 = pd.Variable('b2', data=np.zeros(128))

        # network config
        h1 = pd.relu(pd.matmul(x, W1) + b1)
        fake = pd.matmul(h1, w2) + b2
        prob = pd.sigmoid(fake)
        return prob, fake


theta_D = [D_W1, D_b1, D_W2, D_b2]


def generator(z):
    with pd.Block('generator') as block:
        # declare model parameters
        W1 = pd.get_variable(
            'W1',
            shape=[784, 128],
            initializer=pd.gaussian_random_initializer())
        b1 = pd.get_variable(
            'b1', data=np.zeros(128)
        )  # variable also support initialization using a  numpy data
        W2 = pd.get_variable('W2', data=np.random.rand(128, 1))
        b2 = pd.get_variable('b2', data=np.zeros(128))

        # network config
        h1 = pd.relu(pd.matmul(z, W1) + b1)
        log_prob = pd.matmul(h1, W2) + b2
        prob = pd.sigmoid(log_prob)
        return prob


# a mini-batch of 1. as probability 100%
ones_label = pd.Variable(shape=[None, 1])
# a mini-batch of 0. as probability 0%
zeros_label = pd.Variable(shape=[None, 1])

# model config
G_sample = generator(Z)
D_real_prob, D_real_image = discriminator(X)
D_fake_prob, D_fake_image = discriminator(G_sample)

D_loss_real = pd.reduce_mean(
    pd.cross_entropy(data=D_real_prob, label=ones_label))
D_loss_fake = pd.reduce_mean(
    pd.cross_entropy(data=D_real_fake, label=zeros_label))
D_loss = D_loss_real + D_loss_fake

G_loss = pd.reduce_mean(pd.cross_entropy(data=D_loss_fake, label=ones_label))

D_solver = pd.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = pd.AdamOptimizer().minimize(G_loss, var_list=theta_G)

# init all parameters
initializer = pd.variable_initialzier()
# also ok: initializer = pd.variable_initialzier(vars=theta_D+theta_G)ize,
pd.eval(initializer)


def data_provier(path):
    # ...
    yield batch


for i in range(10000):
    for batch_no, batch in enumerate(data_provider('train_data.txt')):
        # train Descrimator first
        _, D_loss_cur = pd.eval(
            [D_solver, D_loss],
            feeds={
                X: batch,
                Z: sample_Z(batch.size, 10),
                ones_label: np.ones([batch.size, 1]),
                zeros_label: np.zeros([batch.size, 1])
            })
        # get Generator's fake samples
        samples = pd.eval(G_sample, feeds={Z: sample_Z(16, 100)})

        # train Generator latter
        _, G_loss_cur = pd.eval(
            [G_solver, G_loss],
            feeds={
                Z: sample_Z(batch.size, 10),
                ones_label: np.ones([batch.size, 1]),
                zeros_label: np.zeros([batch.size, 1])
            })

        if batch_no % 100:
            logger.info("batch %d, D loss: %f" % (batch_no, D_loss_cur))
            logger.info("batch %d, G loss: %f" % (batch_no, G_loss_cur))
```

