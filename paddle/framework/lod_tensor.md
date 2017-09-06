# Design Doc: LoD (Level-of-Detail) Tensor

PaddlePaddle's RNN doesn't require that all instances have the same length.  To do so, we introduce an extension to Tensor, namely, LoD Tensor.

## Challenge of Variable-length Inputs

People usually represent a mini-batch by a Tensor.  For example, a mini-batch of 32 images, each of size 32x32, is a 10x32x32 Tensor.  So a transformation, T, of all images can be a matrix multiplication of the 32x32xO-dimensional tensor T and the 10x32x32 Tensor.

Another example is that each mini-batch contains 32 sentences, where each word is a D-dimensional one-hot vector.  If all sentences have the same length L, we can represent this mini-batch by a 32xLxD tensor.  However, in most cases, sentences have variable lengths, and we will need an index data structure to record these variable lengths.

## LoD as a Solution

### Mini-Batch of variable-length sentenses

Let's imagine a mini-batch of 3 variable lengths sentences, containing 3, 1, and 2 words respectively.  We can represent it by a (3+1+2)xD tensor plus some index information:

```
   3
3   1 2
||| | ||
```

Each `|` represents a D-dimensional word vectors.  The number 3 on top indicate 3 sentences, and numbers 3, 1, and 2 on the second level represent the number of words in each sentence.

### Mini-Batch of variable-length videos

This approach generalizes to the case where elements are not words, but higher dimensional objects, like images.  Suppose that a mini-batch contains videos of the same frame size 640x480.  If a mini-batch contains 3 videos of 3, 1, and 2 frames respectively.  The underlying tensor is of size (3+1+2)x640x480.  The index information illustrates as:

```
     3
3     1  2
口口口 口 口口
```

where each `口` represents an image.

### Mini-Batch of fixed-size images

Let's get back to a typical example, image classification, where each mini-batch has M fixed-sized images.  The LoD Tensor representation is

```
     M
1 1 1 1     1
口口口口 ... 口
```

The many 1's on the second level seem duplicated.  For this particular case of 2 levels and the second level always have length 1, we can ignore the LoD index.

### Design and summarization

In summary, as long as that the essential elements (words  or images) have the same size, we can represent mini-batches by a LoD Tensor:

- The underlying tensor has size LxD1xD2x..., where D1xD2... is the size of the essential elements, and
- the first dimension size L has an additon property -- a LoD index as a nested vector:

  ```c++
  typedef std::vector<std::vector> > LoD;
  ```

- The LoD index can is not necessary when there are only two levels and all elements of the second level have length 1.

## Slicing of LoD Tensor

Consider that we have a network with three levels of RNN: the top level one handles articles, the second level one handles sentences, and the basic level one handles words.  This network requires that mini-batches represented by 4 level LoD Tensor, for example,

```
         3
3           1  2
3   2  4    1  2  3
||| || |||| |  || |||
```

To allow each level of RNN to handle its input, we define **the slicing of a LoD Tensor is defined as getting the j-th sequence on level i, or the <i,j>-slice**

For example, the <2,1>-slice of above slice is

```
2
||
```

and the <1,2>-slice of above example is

```
2
2  3
|| |||
```

Let's go on slicing this slice.  Its <1,1>-slice is

```
3
|||
```

### The General Slicing Algorithm

The algorithm, with over-simplified data structure, is defined as

```c++
typedef vector<vector<int> > LoD;

struct LoDTensor {
  LoD lod_;
  float* tensor_;
};

LoDTensor Slice(const LoDTensor& lodt, int level, int sequence);
```

### Performance Improvement with Offset
LoD index is crutial to random access performance, for example, a 4 level LoD tensor

```
         3
3           1  2
3   2  4    1  2  3
||| || |||| |  || |||
```
the elements in the upper-level represent the number of elements in the lower-level,
for example, the `3` in the first level means it has three elements `3, 1, 2` in the second level,
the `2` in the second level means it has two elements in the third level `2, 3`.


if we want to access the memory of third element of level 2 (let's name it `elem`), which is a `2`, several operations should be done to get the offset of the tensor memory.

1. sum all the elements ahead `elem` in second level, `3 + 1 = 4`, that means in the third level, there are 4 elements ahead of `elem`.
2. in third level, we get the start offset of `elem` which is `3 + 1 = 4`, and end offset is `3 + 1 + 2 = 6`
3. the forth level is tensor, let's get the start and end offsets, first sum the 4 numbers and get `10`, sum the first 6 numbers and get `15`

finally we get `elem`'s  start and end offset in the tensor level, and access the element's memory.

The above example shows that if the LoD index with number of elements is expensive to access elements's tensor memory.

This can be solved by directly store each element's start offsets from tensor's memory, the uppder LoD can be expressed like

```
0
0             9  10
0   3  5      9  10 12 
```
If we want to get the `elem`'s offset, we can directly get `10` and because the tensor's batch_size is 15, the start and end offsets are directly got without calculation.

the `0` in the first level is useless, so we can delete it and the final format as follows

```
0             9  10
0   3  5      9  10 12 
```
