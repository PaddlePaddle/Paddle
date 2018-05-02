# Varient Length supported RNN Design
For the learning of variable length sequences, the existing mainstream frameworks such as tensorflow, pytorch, caffe2, mxnet and so on all use padding.

Different-length sequences in a mini-batch will be padded with zeros and transformed to same length.

The existing RNN implementations of the PaddlePaddle is `RecurrentLayerGroup`, 
which supports the variable length sequences without padding. 
This doc will design fluid's RNN based on this idea.

## Multi-layer sequence data format `LODTensor`
At present, Paddle stores data in one mini-batch in one-dimensional array.

`Argument.sequenceStartPositions` is used to store information for each sentence.

In Paddle, `Argument.subSequenceStartPositions` is used to store 2 levels of sequence information, while higher dimensional sequences can not be supported.

In order to support the storage of `N-level` sequences, we define sequence information as the following data structure.


```c++
std::shared_ptr<std::vector<std::vector<int>>> lod_start_pos_;
```

Or more clearly defined here

```c++
typedef std::vector<int> level_t;
std::vector<level_t> lod_start_pos;
```
Each `level_t` here stores a level of offset information consistent with paddle's current practice.

In order to transmit sequence information more transparently, we have introduced a new tensor called `LODTensor`[1].
Its tensor-related interfaces all inherit directly from `Tensor`, but it also adds serial-related interfaces.
Thus, when working with a `LODTensor`, ordinary `Op` is used directly as `Tensor`.
The `Op` of the operation sequence will additionally operate the relevant interface of the `LODTensor` variable-length sequence operation.

The definition of `LODTensor` is as follows:


```c++
class LODTensor : public Tensor {
public:
  size_t Levels() const { return seq_start_positions_.size(); }
  size_t Elements(int level = 0) const {
    return seq_start_positions_[level].size();
  }
  // slice of level[elem_begin: elem_end]
  // NOTE low performance in slice seq_start_positions_.
  // TODO should call Tensor's Slice.
  LODTensor LODSlice(int level, int elem_begin, int elem_end) const;

  // slice with tensor's data shared with this.
  LODTensor LODSliceShared(int level, int elem_begin, int elem_end) const;

  // copy other's lod_start_pos_, to share LOD info.
  // NOTE the LOD info sould not be changed.
  void ShareConstLODFrom(const LODTensor &other) {
    lod_start_pos_ = other.lod_start_pos_;
  }
  // copy other's lod_start_pos_'s content, free to mutate.
  void ShareMutableLODFrom(const LODTensor &other) {
    lod_start_pos_ = std::make_shared <
                     std::vector<std::vector<int>>(other.lod_start_pos_.begin(),
                                                   other.lod_start_pos_.end());
  }

private:
  std::shared_ptr<std::vector<std::vector<int>>> lod_start_pos_;
};
```
Among them, `lod_start_pos_` uses `shared_ptr` to reduce the cost of storage and replication.
`LODTensor` can be thought as an extension of `Tensor`, which is almost completely compatible with the original `Tensor`.

## How to support the framework
### Replace `Tensor` with `LoDTensor`
To implement the passing of `LODTensor`, most `Tensor` in the framework need to be replaced with `LODTensor`.
Simple implementation, directly **replace all previous `Tensor` with `LODTensor`** , where you can directly modify the `Tensor` interface created in `pybind.cc`.

In addition, the user may need to perceive the existence of a sequence (such as the sequence of the visualization needs to parse the output sequence in the model), so some of the serial operation APIs also need to be exposed to the python layer.

### Transmit `lod_start_pos` along with the Op call chain
`lod_start_pos` is passed along with the Op call chain
The framework needs to support the following features to implement the transmit of `lod_start_pos`:

1. Implement the transfer as `shared_ptr`
    - Do not modify the contents of `lod_start_pos` as a consumer
    - Modify producer of `lod_start_pos` as producer
    - Conventions consumer only needs to copy `shared_ptr` passed over
    - producer needs to create its own independent memory to store its own independent modifications and expose `shared_ptr` to subsequent consumer
    - Since the transfer process is implemented by copying `shared_ptr`, the framework only needs to pass `lod_start_pos` once.

2. Op is transparent enough not to sense `lod_start_pos`
3. Producer Op that needs to modify `lod_start_pos` can update its `lod_start_pos` data when `Run`

## sorted by length
After sorting by length, the batch size from the forward time step will naturally decrement, and you can directly plug it into Net to do the batch calculation.

For example, the original input:

```
origin:
xxxx
xx
xxx

-> sorted:
xxxx
xxx
xx
```

After `SegmentInputs`, there will be 4 time steps, the input of each time step is as follows (vertical arrangement)

```
0    1    2    3
x    x    x    x
x    x    x
x    x
```

In order to track the changes before and after sorting, use here

```c++
struct SortedSeqItem {
   void *start{nullptr};
   void *end{nullptr};
};

std::vector<SortedSeqItem> sorted_seqs;
```
To track the position of the sequence after sorting, and add a new interface

```c++
std::vector<SortedSeqItem> SortBySeqLen(const LODTensor& tensor);
```
Due to the sequence of input sequences, the following existing interfaces need to be modified:

- InitMemories, memory needs to be rearranged according to `sorted_seqs`
- SetmentInputs
- ConcatOutputs

In addition, because `sorted_seqs` needs to be multiplexed with `RecurrentGradientOp`, it will become a new output of `RecurrentOp`.
It is passed in as an input to `RecurrentGradientOp`.

## InitMemories
Due to the sequence change, the order of the elements on the `boot_memories` batch also needs to be rearranged accordingly.

## SegmentInputs

`SegmentInputs` relies on the information of `sorted_seqs` to cut the original sequence from the horizontal to the input of each step in the sorted sequence order.

the transition is as follows:
```
origin:
xxxx
xx
xxx

   |
   |
  \ /
   !
0    1    2    3
x    x    x    x
x    x    x
x    x
```
## ConcatOutputs
`ConcatOutputs` needs

- Restore the output of each time step back to the original input sequence order (to prevent the order of Infer phase from being upset)
- Concat each sequence as a regular mini-batch representation

## references
1. [Level of details](https://en.wikipedia.org/wiki/Level_of_detail)
