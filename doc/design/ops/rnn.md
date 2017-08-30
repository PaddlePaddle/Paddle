# RNN design
This is the design doc of the recurrent neural network operator, 
and this operator requires that input instances in each mini-batch must have the same length. 

## RNN Algorithm Implementation

<p aligh="center">
<img src="./images/rnn.jpg"/>
</p>

The above diagram shows an RNN being unrolled into a full network.

There are several important concepts:

- stepnet, the network execute in every time step 
- memory, a variable storing state in the current step, which is denoted as $h_t$.
- pre-memory, the value of state in the previous time step, it can be denoted as $h_{t-1}$.
- init-memory, the variable to help initialize state in the first time step.

### step scopes

<p aligh="center">
<img src="./images/rnn.png"/><br/>
fig 2 the RNN's data flow
</p>

Each RNN might run one or more steps, each step runs the same step net.

We use `Scope` to store the contexts of all the step times:

- for each step, create a step Scope
- create all the temporary output variables in the Scope
- execute the step-net, and each step will have its temporary outputs

After all steps finished, RNNOp will collect the specific outputs of each step and merge them to a larger tensor.

### memory and pre-memory
a basic RNN is like:

$$
h_t = U h_{t-1} + W x_t
$$

Here, $h_t$ is time $t$'s state, $h_t$ is time $t-1$'s state, in implementation, we call the a variable that store a state memory.
In step time $t$, $h_t$ is memory, $h_{t-1}$ is pre-memory (short for previous memory).

In each step scope

- each memory variable has a corresponding pre-memory variable
- before a time step executes, copy (or make a reference) the value of previous step scope's memory to the pre-memory variable in current step scope.

### C++ API
- void InferShape(const framework::Scope& scope) const;
  - shape check for inputs and outputs
  - infer the shapes of outputs
  
- void CreateScopes(const framework::Scope& scope) const;
  - create step scopes
  - will be called both in InferShape and Run
- void InitMemories(framework::Scope* step_scopes, bool infer_shape_mode) const;
  - make a reference to the memory in previous step scope and memory in the current one.

- void Run(const framework::Scope& scope, const platform::DeviceContext& dev_ctx) const;
  - run all the time steps.

### User interface
In Paddle's macro design, 
the concept Block represents a sequence of operators.

We can define a RNN's stepnet based on Block as follows

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
        sentence = rnn.segment_input(paragraph)
        h = rnn.memory(shape=[20, 30])
        h.update(
            pd.matmul(W, sentence) + pd.matmul(U, h.pre_state()))
        # get the last state as sentence's info
        h_ = rnn.collect_output(h)
        last_h = pd.lod_last_element(h_)
        return last_h
```
