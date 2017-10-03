# Design for TensorArray
RNN is composed of multiple time steps, in each step, there are several input segments, states and output segments.

## Background
Steps are the core concept of RNNs. In each time step, there should be some input segments, states, and output segments; all the three complements act like arrays, for example, call `states[step_id]` will get the state in `step_id`th time step.

An RNN can be simply implemented with the following pseudo codes

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
According to the [RNN roadmap](https://github.com/PaddlePaddle/Paddle/issues/4561), there are several different RNNs to support.
 
Currently, we have an RNN implementation called `recurrent_op` which takes tensor as input; it splits the input tensors into `input_segments`. 

Considering a tensor can't store variable-length sequences directly, we proposed the tensor with the level of details (`LoDTensor` for short). Segmenting the `LoDTensor` is much more complicated than splitting a tensor, that makes it necessary to refactor the `recurrent_op` with `LoDTensor` segmenting support.

In the second stage, `dynamic_recurrent_op` should be introduced to handle inputs with variable-length sequences. The implementation is the same with `recurrent_op` except that ** how to split the original input `LoDTensors` and outputs to get the `input_segments` and `output_segments`.**. 

In the next stage, a dynamic RNN model based on dynamic operators would be supported. Though it can't be built on `recurrent_op` or `dynamic_recurrent_op` directly, the logic about how to split a tensor or a LoD tensor and get `input_segments` is the same.

## Why `TensorArray`
The three different RNNs may have different logic, but the implementation of how to split the inputs to segments, states and outputs could be shared as a separate module.

The array of `states`, `input_segments` and `output_segments` would be exposed to users when writing a dynamic RNN model similar to the above pseudo codes. 

So there should be an array-like container which might store the segments of a tensor or LoD tensor.

## Introduce TensorArray to uniform all the three RNNs
TensorArray as a new concept is borrowed from TensorFlow, 
it is meant to be used with dynamic iteration primitives such as `while_loop` and `map_fn`.

This concept can be used to support our new design of dynamic operations, and help to refactor some existing variant-sentence-related layers, 
such as `RecurrentGradientMachine`.

In [our design for dynamic RNN](https://github.com/PaddlePaddle/Paddle/pull/4401), 
`TensorArray` is used to segment inputs and store states in all time steps.
By providing some methods similar to a C++ array,
the definition of some state-based dynamic models such as RNN could be more natural and highly flexible.

## Dynamic-Related Methods
Some basic methods should be proposed as follows:

### stack()
Pack the values in a `TensorArray` into a tensor with rank one higher than each tensor in `values`.
### unstack(axis=0)
Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors.
### concat()
Return the values in the `TensorArray` as a concatenated Tensor.
### write(index, value, data_shared=true)
Write value into index of the TensorArray.
### read(index)
Read the value at location `index` in the `TensorArray`.
### size()
Return the number of values.

## LoDTensor-related Supports
The `RecurrentGradientMachine` in Paddle serves as a flexible RNN layer; it takes variant length sequences as input, 
because each step of RNN could only take a tensor-represented batch of data as input, 
some preprocess should be taken on the inputs such as sorting the sentences by their length in descending order and cut each word and pack to new batches.

Such cut-like operations can be embedded into `TensorArray` as general methods called `unpack` and `pack`.

With these two methods, a variant-sentence-RNN can be implemented like

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


some details are as follows.

### unpack(level, sort_by_length)
Split LodTensor in some `level` and generate batches, if set `sort_by_length`, will sort by length.

Returns:

- a new `TensorArray`, whose values are LodTensors and represents batches of data.
- an int32 Tensor, which stores the map from the new batch's indices to original LoDTensor
### pack(level, indices_map)
Recover the original LoD-arranged LoDTensor with the values in a `TensorArray` and `level` and `indices_map`.
