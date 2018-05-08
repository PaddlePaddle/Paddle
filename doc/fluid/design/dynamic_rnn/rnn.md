# RNNOp design

This document describes the RNN (Recurrent Neural Network) operator and how it is implemented in PaddlePaddle. The RNN op requires that all instances in a mini-batch have the same length. We will have a more flexible dynamic RNN operator in the future.

## RNN Algorithm Implementation

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/rnn.jpg"/>
</p>

The above diagram shows an RNN unrolled into a full network.

There are several important concepts here:

- *step-net*: the sub-graph that runs at each step.
- *memory*, $h_t$, the state of the current step.
- *ex-memory*, $h_{t-1}$, the state of the previous step.
- *initial memory value*, the memory of the first (initial) step.

### Step-scope

There could be local variables defined in each step-net.  PaddlePaddle runtime realizes these variables in *step-scopes* which are created for each step.

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/rnn.png"/><br/>
Figure 2 illustrates the RNN's data flow
</p>

Please be aware that every step runs the same step-net.  Each step does the following:

1. Creates the step-scope.
2. Initializes the local variables including step-outputs, in the step-scope.
3. Runs the step-net, which uses the above mentioned variables.

The RNN operator will compose its output from step outputs in each of the step scopes.

### Memory and Ex-memory

Let's give more details about memory and ex-memory using a simple example:

$$
h_t = U h_{t-1} + W x_t
$$,

where $h_t$ and $h_{t-1}$ are the memory and ex-memory (previous memory) of step $t$ respectively.

In the implementation, we can make an ex-memory variable either "refer to" the memory variable of the previous step,
or copy the memory value of the previous step to the current ex-memory variable.

### Usage in Python

For more information on Block, please refer to the [design doc](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/block.md).

We can define an RNN's step-net using a Block:

```python
import paddle as pd

X = some_op() # x is some operator's output and is a LoDTensor
a = some_op()

# declare parameters
W = pd.Variable(shape=[20, 30])
U = pd.Variable(shape=[20, 30])

rnn = pd.create_rnn_op(output_num=1)
with rnn.stepnet():
    x = rnn.add_input(X)
    # declare a memory (rnn's step)
    h = rnn.add_memory(init=a)
    # h.pre_state(), the previous memory of rnn
    new_state = pd.add_two( pd.matmul(W, x) + pd.matmul(U, h.pre_state()))
    # update current memory
    h.update(new_state)
    # indicate that h variables in all step scopes should be merged
    rnn.add_outputs(h)

out = rnn()
```

Python API functions in above example:

- `rnn.add_input`: indicates that the parameter is a variable that will be segmented into step-inputs.
- `rnn.add_memory`: creates a variable used as the memory.
- `rnn.add_outputs`: marks the variables that will be concatenated across steps into the RNN output.

### Nested RNN and LoDTensor

An RNN whose step-net includes other RNN operators is known as an *nested RNN*.

For example, we could have a 2-level RNN, where the top level corresponds to paragraphs, and the lower level corresponds to sentences. Each step of the higher level RNN also receives an input from the corresponding step of the lower level, and additionally the output from the previous time step at the same level.

The following figure illustrates feeding in text into the lower level, one sentence at a step, and the feeding in step outputs to the top level. The final top level output is about the whole text.

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/rnn.png"/>
</p>

```python
import paddle as pd

W = pd.Variable(shape=[20, 30])
U = pd.Variable(shape=[20, 30])

W0 = pd.Variable(shape=[20, 30])
U0 = pd.Variable(shape=[20, 30])

# a is output of some op
a = some_op()

# chapter_data is a set of 128-dim word vectors
# the first level of LoD is sentence
# the second level of LoD is a chapter
chapter_data = pd.Variable(shape=[None, 128], type=pd.lod_tensor, level=2)

def lower_level_rnn(paragraph):
    '''
    x: the input
    '''
    rnn = pd.create_rnn_op(output_num=1)
    with rnn.stepnet():
        sentence = rnn.add_input(paragraph, level=0)
        h = rnn.add_memory(shape=[20, 30])
        h.update(
            pd.matmul(W, sentence) + pd.matmul(U, h.pre_state()))
        # get the last state as sentence's info
        rnn.add_outputs(h)
    return rnn

top_level_rnn = pd.create_rnn_op(output_num=1)
with top_level_rnn.stepnet():
    paragraph_data = rnn.add_input(chapter_data, level=1)
    low_rnn = lower_level_rnn(paragraph_data)
    paragraph_out = low_rnn()

    h = rnn.add_memory(init=a)
    h.update(
        pd.matmul(W0, paragraph_data) + pd.matmul(U0, h.pre_state()))
    top_level_rnn.add_outputs(h)

# output the last step
chapter_out = top_level_rnn(output_all_steps=False)
```

In the above example, the construction of the `top_level_rnn` calls  `lower_level_rnn`.  The input is an LoD Tensor. The top level RNN segments input text data into paragraphs, and the lower level RNN segments each paragraph into sentences.

By default, the `RNNOp` will concatenate the outputs from all the time steps.
If the `output_all_steps` is set to False, it will only output the final time step.


<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/rnn_2level_data.png"/>
</p>
