# RNN 变长输入设计
对变长序列的学习，现有主流框架比如 tensorflow, pytorch, caffe2, mxnet 等均使用了padding的方式，
即将一个mini-batch内不同长度的序列补0到固定长度参与计算。

现有Paddle的 `RecurrentLayerGroup` 实现了无padding的变长序列支持，本文也将基于该模块的思路，设计重构后的变长序列支持。

## 非padding 变长序列的意义
由于tensor必须有明确的shape，因此基于tensor 的主流框架在存储变长序列时，
必须用zero-padding的方式将变长序列补全为固定shape的tensor。

由于padding是一种框架实现变长序列的妥协， 从用户角度，在使用RNN类模型时自然会比较介意padding的存在，
因此会有pytorch中对非padding方式变长序列支持长篇的讨论[3]。

由于padding对内存和计算会有额外的消耗，tensorflow和mxnet均使用了bucketing来就行优化[1][2]，
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
这里为了扩展性，将SeqPosVar定义成如下数据结构来支持N维的序列信息的存储：

```c++
struct SeqPos {
  int dim{1};
  std::vector<std::shared_ptr<std::vector<int>> startPoses;
};
```

其中，startPoses可以用于存储多维的子序列，具体如下：

- 如果为1维序列，则 `dim=1`， `startPoses.size() = 1` 
- 如果为 2 维序列，则 `dim=2`, `startPoses[0]` 存储第一维序列信息，`startPoses[1:]` 存储第二维序列信息
- 如果为 n 维序列，则 `dim=n`, `startPoses[0]` 存储第一维序列，后续追加第 `2.. n` 维序列
  - 当有完整的 n 维序列的 `SeqPos` 信息时，可以从前往后，粒度从粗到细解析序列
  - 当拆解成 n-1 维序列时， `dim=n-1`，startPoses 去除第 1 维序列信息，为每个次级序列单独抽取出对应的信息组成新的 `SeqPos`

Tensor 扩展为
```c++
struct TensorWithSequence {
  Tensor* tensor;
  std::shared_ptr<SeqPos> seq_pos;
}
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

## 参考文献
1. [Tensorflow Bucketing](https://www.tensorflow.org/versions/r0.12/api_docs/python/contrib.training/bucketing)
2. [mxnet Bucketing](http://mxnet.io/how_to/bucketing.html)
3. [variable length input in RNN scenario](https://discuss.pytorch.org/t/about-the-variable-length-input-in-rnn-scenario/345/5)
