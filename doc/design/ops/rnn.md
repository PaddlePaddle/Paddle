# RNNOp design
This is the design doc of the recurrent neural network operator, 
and this operator requires that input instances in each mini-batch must have the same length. 

## RNN Algorithm Implementation

<p aligh="center">
<img src="./images/rnn.jpg"/>
</p>

The above diagram shows an RNN being unrolled into a full network.

There are several important concepts:

- sep-net, the network to be executed in each step
- memory, a variable storing state in the current step, which is denoted as $h_t$.
- pre-memory, the value of state in the previous time step, it can be denoted as $h_{t-1}$.
- init-memory, the variable to help initialize state in the first time step.

### Step Scope
The step-net could have local variables defined.
In each step of RNN scope is created to hold corresponding variables.
Such a scope is known as a *step scope*.

<p aligh="center">
<img src="./images/rnn.png"/><br/>
Figure 2 the RNN's data flow
</p>

All steps run the same step-net.

Each step runs the following procedure:

1. create the step scope,
2. create local variables in the step scope, and
3. execute the step-net, which whould use these variables.

After the execution of all steps, the RNNOp would compose its output from step outputs in those step scopes.

### Memory and Ex-memory
An RNN step often has some status which needs to be passed to the next step.
These status are known as the memory,
which is often referred by the step-net as variables.
A step often needs to refer to the memory value changed by the previous step.
We call it *ex-memory*.

Let's use a simply RNN as an example to explain memory and ex-memory.

$$
h_t = U h_{t-1} + W x_t
$$,

where $h_t$ is the memory of step $t$'s, $h_{t-1}$ is the ex-memory, or the memory of step $t-1$.

In the implementation, we can make an ex-memory variable either "refers to" the memory variable of the previous step, 
or copy the value of the previous memory variable to the current ex-memory variable.

### The Python Interface
In Paddle's macro design, 
the concept Block represents a sequence of operators.

We can define a RNN's step-net based on Block as follows

```python
import paddle as pd

X = some_op() # x is some operator's output, and is a LoDTensor
rnn = pd.RNNOp()

# declare parameters
W = pd.Variable(shape=[20, 30])
U = pd.Variable(shape=[20, 30])

with rnn.stepnet():
    x = rnn.segment_input(X)
    h = rnn.memory(shape=[20, 30], init=pd.gaussion_random_initializer())
    new_state = pd.add_two( pd.matmul(W, x) + pd.matmul(U, h.pre_state()))
    h.update(new_state)
    rnn.collect_output(h)
    
out = rnn()
```

### Integrate with LoDTensor to implement multiple levels of RNN
We are interested in how to implement a multiple levels of RNN, 
for example, use a 2-level RNN to learn the information from a Chapter,
a single chapter contains several paragraphs, 
and each paragraphs contains several setences.

Feed the paragraph data to the 2-th level RNNs, one sentense in each step time, 
collect their outputs and feed to the 2-th level RNN, one paragraph in each step time,
and finally 2-th level RNN outputs the information of a chapter.

<p aligh="center">
<img src="./images/2_level_rnn.png"/>
</p>

```python
import paddle as pd

W = pd.Variable(shape=[20, 30])
U = pd.Variable(shape=[20, 30])

def level2_rnn(paragraph):
    '''
    x: the input
    '''
    rnn = pd.RNNOp()
    with rnn.stepnet():
        sentence = rnn.add_input(paragraph)
        h = rnn.add_memory(shape=[20, 30])
        h.update(
            pd.matmul(W, sentence) + pd.matmul(U, h.pre_state()))
        # get the last state as sentence's info
        rnn.add_output(h)
```
