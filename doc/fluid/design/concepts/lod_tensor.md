# Design Doc: LoD (Level-of-Detail) Tensor

Like other deep learning systems, PaddlePaddle supports training models from sequence data.  Also, like other systems, PaddlePaddle represent a mini-batch of sequences as a Tensor.  What is different is that PaddlePaddle doesn't require all sequences in a mini-batch to be of the same length. Thus no need for padding zeros.

<table>
<thead>
<tr>
<th></th>
<th>TensorFlow</th>
<th>PaddlePaddle</th>
</tr>
</thead>
<tbody>
<tr>
<td>RNN </td>
<td>Support </td>
<td>Support </td>
</tr>
<tr>
<td>recursive RNN </td>
<td>Support </td>
<td>Support </td>
</tr>
<tr>
<td>padding zeros </td>
<td> Must </td>
<td>No need </td>
</tr>
<tr>
<td> blob data type </td>
<td> Tensor</td>
<td> LoDTensor </td>
</tr>
</tbody>
</table>


PaddlePaddle achieves this flexibility by passing through a new data type, *LoD Tensor*, which is a Tensor attached with segmentation index known as *LoD*, between operators.  The LoD index doesn't only segment a tensor, but also recursively segments sub-sequences.  This document presents the design of LoD and LoDTensor.


## The Challenge: Variable-length Sequences

Most deep learning systems represent a mini-batch as a Tensor.  For example, a mini-batch of 10 images, each of size 32x32, is a 10x32x32 Tensor.  Another example is that each mini-batch contains N sentences, where each word is a D-dimensional one-hot vector.  Suppose that all sentences have the same length L, we can represent this mini-batch by a NxLxD tensor.

Both examples show that the elements of sequences are usually of the same size.  In the first example, all images are 32x32, and in the second one, all words are D-dimensional vectors.  It doesn't make sense to allow variable-sized images, as that would require transformations like convolution to handle variable-sized Tensors.

The real challenge is that in most cases, sentences have variable lengths, and we will need an index data structure to segment the tensor into sequences.  Also, sequences might consist of sub-sequences.


## A Solution: The LoD Index

To understand our solution, it is best to look at some examples.

### A Mini-Batch of Sentences

Let's imagine a mini-batch of 3 variable lengths sentences composed of 3, 1, and 2 words, respectively.  We can represent the mini-batch by a (3+1+2)xD tensor plus some index information:

```
3   1 2
||| | ||
```

where each `|` represents a D-dimensional word vector.  The numbers, 3, 1, and 2, form a 1-level LoD.

### Recursive Sequences

Let check another example of a 2-level LoD Tensor.  Consider a mini-batch of three articles with 3, 1, and 2 sentences, and each sentence consists of a variable number of words:

```
3           1  2
3   2  4    1  2  3
||| || |||| |  || |||
```

### A Mini-Batch of Videos

LoD tensors generalize to the case where elements are higher dimensional objects, like images.  Suppose that a mini-batch contains videos of the same frame size 640x480.  Here is a mini-batch of 3 videos with 3, 1, and 2 frames, respectively.

```
3     1  2
口口口 口 口口
```

The underlying tensor is of size (3+1+2)x640x480, and each `口` represents a 640x480 image.

### A Mini-Batch of Images

In traditional cases like a mini-batch with N fixed-sized images,  the LoD Tensor representation is as

```
1 1 1 1     1
口口口口 ... 口
```

In this case, we don't lose any information by ignoring the many 1's in the index and simply considering this LoD Tensor as a usual Tensor:

```
口口口口 ... 口
```

### Model Parameters

A model parameter is just a usual Tensor, which, just like the above example, is a **0-level LoD Tensor**.


## The LoD Tensor

Let us revisit above example of the 2-level LoD Tensor

```
3           1  2
3   2  4    1  2  3
||| || |||| |  || |||
```

It is indeed a tree, where leaves are elementary sequences identified by **branches**.

For example, the third sentence in above example is identified by branch <0,2>, where 0 indicates the first article with length 3, and 2 indicates the third sentence in this article with length 4.

### The LoD Index

We can save the LoD index in the above example

```
3           1  2
3   2  4    1  2  3
```

in a not-full 2D matrix:

```c++
typedef std::vector<std::vector<int> > LoD;
```

where

- `LoD.size()` is the number of levels, or the maximum length of branches,
- `LoD[i][j]` is the length of the j-th segment at the i-th level.

## The Offset Representation

To quickly access elementary sequences, we adopt an offset representation -- instead of saving the lengths, we save the beginning and ending elements of sequences.

In the above example, we accumulate the length of elementary sequences:

```
3 2 4 1 2 3
```

into offsets

```
0  3  5   9   10  12   15
   =  =   =   =   =    =
   3  2+3 4+5 1+9 2+10 3+12
```

so we know that the first sentence is from word 0 to word 3, and the second sentence from word 3 to word 5.

Similarly, the lengths in the top level LoD

```
3 1 2
```

are transformed into offsets of elements/words as follows:

```
0 3 4   6
  = =   =
  3 3+1 4+2
```

## Slicing of LoD Tensors


When we use the above 2-level LoD Tensor as the input to a nested-RNN, we need to retrieve certain sequences.  Here we define the sequence identified by branch <i,j,...> as the **<i,j,...>-slice**.

For example, the <2>-slice of above example is

```
10      15
10  12  15
  || |||
```

and the <2,0>-slice of above slice is

```
10  12
  ||
```

## Length Representation vs Offset Representation

The offset representation is an implementation-oriented decision and it makes understanding the idea behind LoDTensor difficult.
Hence, we encapsulate this implementation detail in C++ and expose the original length representation in our Python API. 
Specifically, we call this length representation `recursive_sequence_lengths` and users can use the following code to set or get the `recursive_sequence_lengths` of a LoDTensor in Python:
```Python
# length representation of lod called recursive_sequence_lengths
recursive_seq_lens = [[3, 1, 2], [2, 2, 1, 3, 1, 2]]
# Create a LoDTensor that has the above recursive_sequence_lengths info.
# This recursive_sequence_lengths will be converted to an offset representation of LoD in the C++ implementation under the hood.
tensor = fluid.LoDTensor(lod)

# Set/Change the recursive_sequence_lengths info of LoDTensor
tensor.set_recursive_sequence_lengths([[3, 1, 2]])
# Get the recursive_sequence_lengths info of a LoDTensor (the offset-based LoD representation stored in C++ will be converted 
# back to length-based recursive_sequence_lengths), new_recursive_seq_lens = [[3, 1, 2]]
new_recursive_seq_lens = tensor.recursive_sequence_lengths()
```
