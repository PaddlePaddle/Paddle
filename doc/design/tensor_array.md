# Design for TensorArray
This design doc presents the necessity of a new C++ class `TensorArray`.
In addition to the very simple C++ implementation

```c++
class TensorArray {
 public:
  explicit TensorArray(const LoDTensor&);
  explicit TensorArray(size_t size);

 private:
  vector<LoDTensor> values_;
};
```

We also need to expose it to PaddlePaddle's Python API,
because users would want to use it with our very flexible operators `WhileLoop`.
An example for a RNN based on dynamic operators is 

```python
input = pd.data(...)
num_steps = Var(12)

TensorArray states(size=num_steps)
TensorArray step_inputs(unstack_from=input)
TensorArray step_outputs(size=num_steps)

W = Tensor(...)
U = Tensor(...)
default_state = some_op()

step = Var(1)

wloop = paddle.create_whileloop(loop_vars=[step])
with wloop.frame():
    wloop.break_if(pd.equal(step, num_steps)
    pre_state = states.read(step-1, default_state)
    step_input = step_inputs.read(step)
    state = pd.sigmoid(pd.matmul(U, pre_state) + pd.matmul(W, step_input))
    states.write(step, state)
    step_outputs.write(step, state) # output state
    step.update(state+1)

output = step_outputs.stack()
```

## Background
Steps are one of the core concepts of RNN. In each time step of RNN, there should be several input segments, states, and output segments; all these components act like arrays, for example, call `states[step_id]` will get the state in `step_id`th time step.

An RNN can be implemented with the following pseudocode

```c++
Array states;
Array input_segments;
Array output_segments;
Parameter W, U;

step = 1
seq_len = 12
while_loop {
   if (step == seq_len) break;
    states[step] = sigmoid(W * states[step-1] + U * input_segments[step]);
    output_segments[step] = states[step] // take state as output
   step++;
}
```
According to the [RNN roadmap](https://github.com/PaddlePaddle/Paddle/issues/4561), there are several different RNNs that PaddlePaddle will eventually support.

Currently, the basic RNN implementation supported by PaddlePaddle is the `recurrent_op` which takes tensors as input and splits them into `input_segments`.


Since a tensor cannot store variable-length sequences directly, PaddlePaddle implements the tensor with level of details (`LoDTensor` for short).
Segmenting the `LoDTensor` is much more complicated than splitting a tensor, that makes it necessary to refactor the `recurrent_op` with `LoDTensor` segmenting support.

As the next step in RNN support, `dynamic_recurrent_op` should be introduced to handle inputs with variable-length sequences.

The implementation is similar to `recurrent_op`. 
The key difference is the way **the original input `LoDTensors` and outupts are split to get the `input_segments` and the `output_segments`.**


Though it can't be built over `recurrent_op` or `dynamic_recurrent_op` directly,
the logic behind splitting a tensor or a LoD tensor into `input_segments` remains the same.

## Why `TensorArray`
The logic behind splitting the inputs to segments, states and outputs is similar and can be shared in a seperate module.

The array of `states`, `input_segments` and `output_segments` would be exposed to users when writing a dynamic RNN model similar to the above pseudo codes. 

So there should be an array-like container, which can store the segments of a tensor or LoD tensor.

**This container can store an array of tensors and provides several methods to split a tensor or a LoD tensor** .
This is where the notion of `TensorArray` comes from.

## Introduce TensorArray to uniform all the three RNNs
TensorArray as a new concept is borrowed from TensorFlow, 
it is meant to be used with dynamic iteration primitives such as `while_loop` and `map_fn`.

This concept can be used to support our new design of dynamic operations, and help to refactor some existing variant-sentence-related layers, 
such as `recurrent_op`, `RecurrentGradientMachine`.

In [our design for dynamic RNN](https://github.com/PaddlePaddle/Paddle/pull/4401), 
`TensorArray` is used to segment inputs and store states in all time steps.
By providing some methods similar to a C++ array,
the definition of some state-based dynamic models such as RNN can be more natural and highly flexible.

## Dynamic-operations on TensorArray

`TensorArray` will be used directly when defining dynamic models, so some operators listed below should be implemented

```python
# several helper operators for TensorArray
def tensor_array_stack(ta, tensor):
    '''
    get a tensor array `ta`, return a packed `tensor`.
    '''
    pass

def tensor_array_unstack(tensor, ta):
    '''
    get a `tensor`, unstack it and get a tensor array `ta`.
    '''
    pass

def tensor_array_write(ta, index, tensor, data_shared):
    '''
    get a `tensor` and a scalar tensor `index`, write `tensor` into index-th
    value of the tensor array `ta`.
    `data_shared` is an attribute that specifies whether to copy or reference the tensors.
    '''
    pass

def tensor_array_read(ta, index, tensor):
    '''
    get a tensor array `ta`, a scalar tensor `index`, read the index-th value of
    `ta` and return as the `tensor`.
    '''
    pass

def tensor_array_size(ta, tensor):
    '''
    get a tensor array `ta`, return the size of `ta` and return as the scalar `tensor`.
    '''
    pass
```

It is trivial for users to use so many low-level operators, so some helper methods should be proposed in python wrapper to make `TensorArray` easier to use, 
for example

```python
class TensorArray:
    def __init__(self, name):
        self.name = name
        self.desc = TensorArrayDesc()

    def stack(self, name=None):
        '''
        Pack the values in a `TensorArray` into a tensor with rank one higher
        than each tensor in `values`.
        `stack` can be used to split tensor into time steps for RNN or whileloop.

        @name: str
            the name of the variable to output.
        '''
        tensor = Var(name)
        tensor_array_stack(self.name, tensor)
        return tensor

    def unstack(self, input):
        '''
        Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors.
        `unstack` can be used to concatenate all the time steps for RNN or whileloop.

        @input: str
            the name of input tensor
        '''
        tensor_array_unstack(tensor, self.name)

    def write(self, index, value, data_shared=True):
        '''
        Write value into index of the TensorArray.
        If `data_shared` is set to True, than the index-th value in TensorArray will
        be shared with the tensor passed in.

        @index: str
            name of a scalar tensor
        @value: str
            name of a tensor
        @data_shared: bool
        '''
        tensor_array_write(self.name, index, value, data_shared)

    def read(self, index, output):
        '''
        Read the value at location `index` in the `TensorArray`.

        @index: str
            name of a scalar tensor
        @output:
            name of a output variable
        '''
        tensor_array_read(self.name, index, output)


    def size(self, output):
        '''
        Return the number of values.

        @output: str
            name of a scalar tensor
        '''
        tensor_array_size(self.name, output)
```

## LoDTensor-related Supports
The `RecurrentGradientMachine` in Paddle serves as a flexible RNN layer; it takes varience-length sequences as input, and output sequences too.

Since each step of RNN can only take a tensor-represented batch of data as input, 
some preprocess should be taken on the inputs such as sorting the sentences by their length in descending order and cut each word and pack to new batches.

Such cut-like operations can be embedded into `TensorArray` as general methods called `unpack` and `pack`,
these two operations are similar to `stack` and `unstack` except that they operate on variable-length sequences formated as a LoD tensor rather than a tensor.

Some definitions are like

```python
def unpack(level):
    '''
    Split LodTensor in some `level` and generate batches, if set `sort_by_length`,
    will sort by length.

    Returns:
        - a new `TensorArray`, whose values are LodTensors and represents batches
          of data.
        - an int32 Tensor, which stores the map from the new batch's indices to
          original LoDTensor
    '''
    pass

def pack(level, indices_map):
    '''
    Recover the original LoD-arranged LoDTensor with the values in a `TensorArray`
    and `level` and `indices_map`.
    '''
    pass
```

With these two methods, a varience-length sentence supported RNN can be implemented like

```c++
// input is the varient-length data
LodTensor sentence_input(xxx);
TensorArray ta;
Tensor indice_map;
Tensor boot_state = xxx; // to initialize rnn's first state
TensorArray::unpack(input, 1/*level*/, true/*sort_by_length*/, &ta, &indice_map);
TessorArray step_outputs;
TensorArray states;

for (int step = 0; step = ta.size(); step++) {
  auto state = states.read(step);
  // rnnstep is a function which acts like a step of RNN
  auto step_input = ta.read(step);
  auto step_output = rnnstep(step_input, state);
  step_outputs.write(step_output, true/*data_shared*/);
}

// rnn_output is the final output of an rnn
LoDTensor rnn_output = ta.pack(ta, indice_map);
```
the code above shows that by embedding the LoDTensor-related preprocess operations into `TensorArray`,
the implementation of a RNN that supports varient-length sentences is far more concise than `RecurrentGradientMachine` because the latter mixes all the codes together, hard to read and extend.
