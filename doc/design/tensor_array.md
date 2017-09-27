# Design for TensorArray
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
TensorArray::unpack(input, 1/*level*/, true/*sort_by_length*/, &ta, &indice_map);
TessorArray step_outputs;

for (int step = 0; step = ta.size(); step++) {
  // step_output is a tensor
  // rnnstep is a function which acts like a step of RNN
  auto step_output = rnnstep(ta.read(step))
  step_outputs.write(step_output, true/*data_shared*/);
}

// rnn_output is the final output of an rnn
LoDTensor rnn_output;
pack(ta, indice_map, &rnn_output);
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
