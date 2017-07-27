# RNN 变长输入设计
对变长序列的学习，现有主流框架比如 tensorflow, pytorch, caffe2, mxnet 等均使用了padding的方式，
即将一个mini-batch内不同长度的序列补0到固定长度参与计算。

现有Paddle包括 `RecurrentLayerGroup` 在内的RNN均实现了无padding的变长序列支持，本文也将基于该模块的思路，设计重构后的变长序列支持。

## 非padding 变长序列的意义
由于tensor必须有明确的shape，因此基于tensor 的主流框架在存储变长序列时，
必须用zero-padding的方式将变长序列补全为固定shape的tensor。

由于padding是一种框架实现变长序列的妥协， 从用户角度，在使用RNN类模型时自然会比较介意padding的存在，
因此会有pytorch中对非padding方式变长序列支持长篇的讨论[3]。

由于padding对内存和计算会有额外的消耗，tensorflow和mxnet均使用了bucketing来进行优化[1][2]，
但不管是padding还是bucket，对于用户都是额外的使用负担。

因此，**paddle原生支持变长序列的方式，能直接满足用户对变长序列的最直接的需求，在当前主流平台中可以算是一大优势**。

但对变长序列的支持，需要对目前框架做一些修改，下面讨论如何在最小修改下支持变长序列。

## 变长数据格式
目前 Paddle 会将一个mini-batch内的数据存储在一维的内存上，
额外使用 `Argument.sequenceStartPositions` 来存储每个句子的信息。

基于当前重构现状，我们使用如下设计来存储变长数据格式

- 扩充 Tensor 以支持存储变长序列的信息（这部分信息后续用SeqPosVar表示）
- Op 的 `InferShape` 会更新outputs 的`SeqPosVar`
- 为了兼容序列Op（比如RNN）和传统Op（比如FC），序列的所有元素均flatten追加存储到一个mini-batch中
  - 比如，长度分别为2,3,4的三个句子会存储为一个size为9的`mini-batch`
  - 额外会有一个`SeqPosVar`，存储句子的结构，比如offest：`0,2,5,9`
  
为了支持sub-sequence，Paddle里使用 `Argument.subSequenceStartPositions` 来存储2维的序列信息，更高维度的序列无法支持；
这里为了扩展性，将SeqPosVar定义成如下数据结构来支持N维的序列信息的存储

```c++
std::vector <std::vector<std::vector<int>> seq_start_positions_;
```

附录中演示如何用二维的vector来存储多个 level 的变长序列的start position.

Tensor 扩展为
```c++
/*
 * Tensor storing sequences.
 */
class TensorWithSequence {
public:
  Tenser *tensor() { return tensor_; }

  /*
   * get an element of current level.
   */
  TensorWithSequence Element(int element) const;

  /*
   * get an element of n-th level.
   * NOTE low performance.
   */
  TensorWithSequence Element(int level, int element) const;

  /*
   * get number of elements in n-th level.
   */
  size_t Elements(int level = 0) const;

  /*
   * get the number of levels of sequences.
   */
  size_t Levels() const;

  /*
   * copy other's pointers to share their data.
   */
  void ShareDataFrom(const TensorWithSequence &other);

  /*
   * just copy other's sequence info (use shared_ptr to share memory).
   */
  void ShareSeqPosFrom(const TensorWithSequence &other);

  /*
   * copy others' sequence info for mutation.
   */
  void CopySeqPosFrom(const TensorWithSequence &other);

private:
  Tensor *tensor_;
  /*
   * store start positions of all levels.
   *
   * data format like
   *
   *   0-th level start positions
   *   1-th level, element 0, start positions
   *   1-th level, element 1, start positions
   *   ...
   *   1-th level, element k, start positions
   *   2-th level, element 0, start positions
   *   2-th level, element 1, start positions
   *   ...
   *   2-th level, element n, start positions
   *   ...
   *
   */
  std::vector < std::vector<std::vector<int>> seq_start_positions_;
};
```

## 框架支持方法
类似Paddle现在的做法，为了支持每个参与inputs/outputs的variable必须有对应的SeqPosVar，
**这里需要框架就行一些修改，有一些trick的成分**。

现有框架可以在 `Context` 里添加一个与 `Input` 平行的接口 `InputSeq` 来获取序列信息，具体定义如下

```
std::shared_ptr<SeqPos> InputSeq(const std::string& name);
```

为了能够将SeqPos在Op的调用关系中传递下去，考虑到一些不支持序列的Op（比如FC）可能丢失SeqPos，
框架需要强制所有的OP的InferShape都必须感知并传递SeqPos，
目前最简单的方式是直接在 OperatorBase的InferShape里设置

```c++
void InferShape(const std::shared_ptr<Scope<>& scope) {
  CopyInSeqToOut();
  // ...
}

// if inputs has SeqPos, copy to output.
void CopyInSeqToOut();
```

## 根据长度排序
按照长度排序后，从前往后的时间步的batch size会自然地递减，这是 Net 支持的

比如：

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

经过 `SegmentInputs` 之后，每个会有4个时间步，每个时间步的输入如下（纵向排列）

```
0    1    2    3
x    x    x    x
x    x    x
x    x
```

为了追踪排序前后序列的变化，这里用
```c++
struct SortedSeqItem {
   void *start{nullptr};
   void *end{nullptr};
};

std::vector<SortedSeqItem> sorted_seqs;
```
来追踪序列排序后的位置。

对比现有设计，只需要修改 `InitMemories`, `SegmentInputs` 和 `ConcatOutputs` 两个接口，此外添加一个 `SortBySeqLen` 的接口，
就可以支持上述变长序列，下面详细介绍。
## InitMemories
由于序列顺序的变化，`boot_memories` 的batch上的element的顺序也需要对应重新排列。

## SegmentInputs
`SegmentInputs` 会依赖 `sorted_seqs` 的信息，将原始的序列按照排序后的序列顺序，从横向切割，转为每个step中的inputs。

即下面的转变：
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
`ConcatOutputs` 需要

- 将每个时间步的输出重新还原为原始输入的序列顺序（以防止Infer阶段顺序打乱）
- 将每个序列concat 为规则的mini-batch表示

## 附录
这里演示多level的变长序列的存储方法，本设计会用两层的`vector` 来存储所有序列的信息，具体数据格式如下

```c++
std::vector < std::vector<std::vector<int>> seq_start_positions_;
```
为了方便讨论，可以临时修改为
```c++
typedef std::vector<int> element_t;
std::vector<element_t> seq_start_positions_;
```

假设tensor 里按batch存储 instance作为基本单位， 
默认序列里的元素都是相邻排列，
因此只需要以instance 为基本单位，
记录 start position就可以分解出每个序列的信息。

`seq_start_positions_` 里从上往下存储着 `level 0 ~ level L`的元素，可以认为level越小，表示的序列粒度越大。
比如存储 `batch of paragraphs` 则有

- `level 0` 存储 paragraphs 的 start positions 
- `level 1` 存储 sentences 的 start positions 

因为 tensor 里存储着batch of words，所以以上两个level的start positions的单位均为word。

具体地，假设有如下例子，比如需要存储 batch of paragraphs，tensor中存储了 batch of words，而序列信息如下

- paragraph 0 has 3 sentences:
  - sentence 0 has 3 words
  - sentence 1 has 4 words
  - sentence 2 has 2 words
- paragraph 1 has 2 sentences:
  - sentence 0 has 5 words
  - sentence 1 has 3 words

那么`seq_start_positions_` 会有如下内容

- 0 9(=3+4+2)
- 0 3 7
- 0 5

其中每行是一个 `element_t`，具体含义如下

- `seq_start_positions_[0]` 存储了`0 9` ，表示paragraph 0 在 tensor 中的偏移为 0，对应地， paragraph 1 为 9 (以word 为单位)
- 从 `seq_start_positions_[0]` 中可以知道，当前 `mini-batch` 总共只有 2 个 paragraph，因此后续的两个 `element_t` 分别存储了两个 paragraph 中句子的信息
- 紧接着`seq_start_positions_[1]` 存储了第0个paragraph 的信息，表明有3个sentence，其在paragraph 0在tensor中对应部分的偏移分别为0,3 和7
- 紧接着`seq_start_positions_[2]` 存储了第1个paragraph 的信息，表明有2个sentence，其在paragraph 0在tensor中对应部分的偏移分别为0和 5

如上证明了`seq_start_positions_`的数据结构适用于 level 为 1（也就是Paddle中subseq）， **通过归纳法可以证明其适用于 N level 的序列，这里暂不赘述** 。

## 参考文献
1. [Tensorflow Bucketing](https://www.tensorflow.org/versions/r0.12/api_docs/python/contrib.training/bucketing)
2. [mxnet Bucketing](http://mxnet.io/how_to/bucketing.html)
3. [variable length input in RNN scenario](https://discuss.pytorch.org/t/about-the-variable-length-input-in-rnn-scenario/345/5)
