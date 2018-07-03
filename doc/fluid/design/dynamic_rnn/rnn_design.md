# RNN 变长输入设计
对变长序列的学习，现有主流框架比如 tensorflow, pytorch, caffe2, mxnet 等均使用了padding的方式，
即将一个mini-batch内不同长度的序列补0到固定长度参与计算。

现有Paddle包括 `RecurrentLayerGroup` 在内的RNN均实现了无padding的变长序列支持，本文也将基于该模块的思路，设计重构后的变长序列支持。

## 背景介绍
由于tensor必须有明确的shape，因此基于tensor 的主流框架在存储变长序列时，
必须用zero-padding的方式将变长序列补全为固定shape的tensor。

由于padding是一种框架实现变长序列的妥协， 从用户角度，在使用RNN类模型时自然会比较介意padding的存在，
因此会有pytorch中对非padding方式变长序列支持长篇的讨论[3]。

由于padding对内存和计算会有额外的消耗，tensorflow和mxnet均使用了bucketing来进行优化[1][2]，
但不管是padding还是bucket，对于用户都是额外的使用负担。

因此，**paddle原生支持变长序列的方式，能直接满足用户对变长序列的最直接的需求，在当前主流平台中可以算是一大优势**。

但对变长序列的支持，需要对目前框架做一些修改，下面讨论如何在最小修改下支持变长序列。

## 多层序列数据格式 `LODTensor`
目前 Paddle 会将一个mini-batch内的数据存储在一维的内存上，
额外使用 `Argument.sequenceStartPositions` 来存储每个句子的信息。

Paddle里使用 `Argument.subSequenceStartPositions` 来存储2层的序列信息，更高维度的序列则无法直接支持；

为了支持 `N-level` 序列的存储，本文将序列信息定义成如下数据结构:

```c++
std::shared_ptr<std::vector<std::vector<int>>> lod_start_pos_;
```

或者更明确的定义

```c++
typedef std::vector<int> level_t;
std::vector<level_t> lod_start_pos;
```

这里的每一个 `level_t` 存储一个粒度(level)的偏移信息，和paddle目前做法一致。

为了更透明地传递序列信息，我们引入了一种新的tensor 称为 `LODTensor`[4]，
其关于tensor相关的接口都直接继承自 `Tensor`，但另外添加了序列相关接口。
如此，在操作一个 `LODTensor` 时，普通 `Op` 直接当成 `Tensor` 使用，
而操作序列的 `Op` 会额外操作 `LODTensor` 的变长序列操作的相关接口。

`LODTensor` 具体定义如下：

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

其中， `lod_start_pos_` 使用了 `shared_ptr` 来减少存储和复制的代价，
可以认为 `LODTensor` 是 `Tensor` 的扩展，几乎完全兼容原始 `Tensor` 的使用。

## 框架支持
### 框架现有的 `Tensor` 调用替换为 `LODTensor`
为了实现 `LODTensor` 的传递，框架里很多 `Tensor` 都需要变成 `LODTensor`，
简单实现，直接 **把之前所有的`Tensor` 全部替换成 `LODTensor`，这里可以直接修改 `pybind.cc` 里面创建`Tensor`的接口**。

此外，用户有可能需要感知序列的存在（比如序列的可视化需要解析模型中输出的序列），因此一些序列操作的API也需要暴露到 python 层。

### `lod_start_pos` 随着Op调用链传递
框架需要支持下列特性，以实现`lod_start_pos`的传递：

1. 以 `shared_ptr` 的方式实现传递
    - 不修改 `lod_start_pos` 内容的作为 consumer
    - 修改 `lod_start_pos` 的作为 producer
    - 约定 consumer 只需要复制传递过来的 `shared_ptr`
      - producer 需要创建自己的独立的内存，以存储自己独立的修改，并暴露 `shared_ptr` 给后续 consumer
    - 由于传递过程是以复制`shared_ptr`的方式实现，因此框架只需要传递一次 `lod_start_pos`

2. 对于不感知 `lod_start_pos` 的Op足够透明
3. 需要修改 `lod_start_pos` 的producer Op可以在 `Run` 时更新自己的 `lod_start_pos` 数据

具体的设计分为以下3小节

#### `load_start_pos` 的传递

- 对于不需要修改 `lod_start_pos` 的情况，调用 LODTensor的 `ShareConstLODFrom` 接口实现复制
- 需要修改的，调用`ShareMutableLODFrom` 接口自己分配内存以存储修改

#### 框架透明
传递这一步需要加入到网络跑之前的初始化操作中，并且只需要初始化一次，基于当前框架设计的初步方案如下

- 在 Op 的 `attrs` 中添加一项 `do_mutate_lod_info` 的属性，默认为 `false`
  - 有需要修改 `lod_start_pos` 的Op需要在定义 `OpProto` 时设置为 `true`
- `OperatorBase` 的 `InferShape` 中会读取 `do_mutate_lod_info` ，并且调用 `LODTensor` 相关的方法实现 `lod_start_pos` 的复制。
- `OperatorBase` 中添加一个 member `is_lod_inited{false}` 来保证传递只进行一次

一些逻辑如下

```c++
class OperatorBase {
public:
  // ...
  void InferShape() {
    if (!is_load_inited) {
      bool do_mutate_lod_info = GetAttr<bool>("do_mutate_load_info");
      // find a input having LOD to copy
      auto lod_input = ValidLODInput();
      for (auto &output : outputs) {
        if (do_mutate_load_info) {
          output.ShareMutableLODFrom(lod_input);
        } else {
          output.ShareConstLODFrom(load_input);
        }
      }
      is_pod_inited = true;
    }

    // call op's InferShape
    // ...
  }

private:
  // ...
  bool is_lod_inited{false};
};
```

如此，`lod_start_pos` 的信息的传递对非OLD的Op的实现是完全透明的。

#### `lod_start_pos` 的更新
上一小节介绍到，对于需要修改 `load_start_pos` 的Op，`OperatorBase` 会分配一块自己的内存以存储修改，
Op在 `Run` 的实现中，操作更新自己的 `load_start_pos` ，
而所有依赖其 outputs 的 op 会通过共享的指针自动获取到其更新。

## 根据长度排序
按照长度排序后，从前往后的时间步的batch size会自然地递减，可以直接塞入 Net 做batch计算

比如原始的输入：

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
来追踪序列排序后的位置，并添加一个新的接口

```c++
std::vector<SortedSeqItem> SortBySeqLen(const LODTensor& tensor);
```

由于输入序列的顺序变化，以下现有的接口需要针对性地修改：

- InitMemories, memory需要根据 `sorted_seqs` 重新排列
- SetmentInputs
- ConcatOutputs

此外，由于 `sorted_seqs` 需要被 `RecurrentGradientOp` 复用，因此会变成 `RecurrentOp` 一个新的output输出，
之后作为 `RecurrentGradientOp` 的一个输入传入。

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

## 参考文献
[Tensorflow Bucketing](https://www.tensorflow.org/versions/r0.12/api_docs/python/contrib.training/bucketing)

[mxnet Bucketing](http://mxnet.io/how_to/bucketing.html)

[variable length input in RNN scenario](https://discuss.pytorch.org/t/about-the-variable-length-input-in-rnn-scenario/345/5)

[Level of details](https://en.wikipedia.org/wiki/Level_of_detail)
