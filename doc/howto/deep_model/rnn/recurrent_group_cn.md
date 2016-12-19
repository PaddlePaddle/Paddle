# Recurrent Group教程

## 概述

序列数据是自然语言处理任务面对的一种主要输入数据类型。

一句话是由词语构成的序列，多句话进一步构成了段落。因此，段落可以看作是一个嵌套的双层的序列，这个序列的每个元素又是一个序列。

双层序列是PaddlePaddle支持的一种非常灵活的数据组织方式，帮助我们更好地描述段落、多轮对话等更为复杂的语言数据。基于双层序列输入，我们可以设计搭建一个灵活的、层次化的RNN，分别从词语和句子级别编码输入数据，同时也能够引入更加复杂的记忆机制，更好地完成一些复杂的语言理解任务。

在PaddlePaddle中，`recurrent_group`是一种任意复杂的RNN单元，用户只需定义RNN在一个时间步内完成的计算，PaddlePaddle负责完成信息和误差在时间序列上的传播。

更进一步，`recurrent_group`同样可以扩展到双层序列的处理上。通过两个嵌套的`recurrent_group`分别定义子句级别和词语级别上需要完成的运算，最终实现一个层次化的复杂RNN。

目前，在PaddlePaddle中，能够对双向序列进行处理的有`recurrent_group`和部分Layer，具体可参考文档：<a href = "hierarchical_layer_cn.html">支持双层序列作为输入的Layer</a>。
 
## 相关概念

### 基本原理
`recurrent_group` 是PaddlePaddle支持的一种任意复杂的RNN单元。使用者只需要关注于设计RNN在一个时间步之内完成的计算，PaddlePaddle负责完成信息和梯度在时间序列上的传播。

PaddlePaddle中，`recurrent_group`的一个简单调用如下：

``` python
recurrent_group(step, input, reverse)
```
- step：一个可调用的函数，定义一个时间步之内RNN单元完成的计算
- input：输入，必须是一个单层序列，或者一个双层序列
- reverse：是否以逆序处理输入序列
 
使用`recurrent_group`的核心是设计step函数的计算逻辑。step函数内部可以自由组合PaddlePaddle支持的各种layer，完成任意的运算逻辑。`recurrent_group` 的输入（即input）会成为step函数的输入，由于step 函数只关注于RNN一个时间步之内的计算，在这里`recurrent_group`替我们完成了原始输入数据的拆分。

### 输入
`recurrent_group`处理的输入序列主要分为以下三种类型：
 
- **数据输入**：一个双层序列进入`recurrent_group`会被拆解为一个单层序列，一个单层序列进入`recurrent_group`会被拆解为非序列，然后交给step函数，这一过程对用户是完全透明的。可以有以下两种：1）通过data_layer拿到的用户输入；2）其它layer的输出。
		
- **只读Memory输入**：`StaticInput` 定义了一个只读的Memory，由`StaticInput`指定的输入不会被`recurrent_group`拆解，`recurrent_group` 循环展开的每个时间步总是能够引用所有输入，可以是一个非序列，或者一个单层序列。
	  
- **序列生成任务的输入**：`GeneratedInput`只用于在序列生成任务中指定输入数据。

### 输入示例

序列生成任务大多遵循encoder-decoer架构，encoder和decoder可以是能够处理序列的任意神经网络单元，而RNN是最流行的选择。

给定encoder输出和当前词，decoder每次预测产生下一个最可能的词语。在这种结构中，decoder接受两个输入：
    
- 要生成的目标序列：是decoder的数据输入，也是decoder循环展开的依据，`recurrent_group`会对这类输入进行拆解。

- encoder输出，可以是一个非序列，或者一个单层序列：是一个unbounded memory，decoder循环展开的每一个时间步会引用全部结果，不应该被拆解，这种类型的输入必须通过`StaticInput`指定。关于Unbounded Memory的更多讨论请参考论文 [Neural Turning Machine](https://arxiv.org/abs/1410.5401)。
		
在序列生成任务中，decoder RNN总是引用上一时刻预测出的词的词向量，作为当前时刻输入。`GeneratedInput`自动完成这一过程。
		 
### 输出
`step`函数必须返回一个或多个Layer的输出，这个Layer的输出会作为整个`recurrent_group` 最终的输出结果。在输出的过程中，`recurrent_group` 会将每个时间步的输出拼接，这个过程对用户也是透明的。

### memory
memory只能在`recurrent_group`中定义和使用。memory不能独立存在，必须指向一个PaddlePaddle定义的Layer。引用memory得到这layer上一时刻输出，因此，可以将memory理解为一个时延操作。

可以显示地指定一个layer的输出用于初始化memory。不指定时，memory默认初始化为0。

## 双层RNN介绍
`recurrent_group`帮助我们完成对输入序列的拆分，对输出的合并，以及计算逻辑在序列上的循环展开。

利用这种特性，两个嵌套的`recurrent_group`能够处理双层序列，实现词语和句子两个级别的双层RNN结构。

- 单层（word-level）RNN：每个状态（state）对应一个词（word）。
- 双层（sequence-level）RNN：一个双层RNN由多个单层RNN组成，每个单层RNN（即双层RNN的每个状态）对应一个子句（subseq）。

为了描述方便，下文以NLP任务为例，将含有子句（subseq）的段落定义为一个双层序列，将含有词语的句子定义为一个单层序列，那么0层序列即为一个词语。

## 双层RNN的使用

### 训练流程的使用方法
使用 `recurrent_group`需要遵循以下约定：
 
- **单进单出**：输入和输出都是单层序列。
  - 如果有多个输入，不同输入序列含有的词语数必须严格相等。
  - 输出一个单层序列，输出序列的词语数和输入序列一致。
  - memory：在step函数中定义 memory指向一个layer，通过引用memory得到这个layer上一个时刻输出，形成recurrent 连接。memory的is_seq参数必须为false。如果没有定义memory，每个时间步之内的运算是独立的。
  - boot_layer：memory的初始状态，默认初始状为0，memory的is_seq参数必须为false。
 
- **双进双出**：输入和输出都是双层序列。
  - 如果有多个输入序列，不同输入含有的子句（subseq）数必须严格相等，但子句含有的词语数可以不相等。
  - 输出一个双层序列，子句（subseq）数、子句的单词数和指定的一个输入序列一致，默认为第一个输入。
  - memory：在step函数中定义memory，指向一个layer，通过引用memory得到这个layer上一个时刻的输出，形成recurrent连接。定义在外层`recurrent_group` step函数中的memory，能够记录上一个subseq 的状态，可以是一个单层序列（只作为read-only memory），也可以是一个词语。如果没有定义memory，那么 subseq 之间的运算是独立的。
  - boot_layer：memory 初始状态，可以是一个单层序列（只作为read-only memory）或一个向量。默认不设置，即初始状态为0。

- **双进单出**：目前还未支持，会报错"In hierachical RNN, all out links should be from sequences now"。
 

### 生成流程的使用方法
使用`beam_search`需要遵循以下约定：

- 单层RNN：从一个word生成下一个word。
- 双层RNN：即把单层RNN生成后的subseq给拼接成一个新的双层seq。从语义上看，也不存在一个subseq直接生成下一个subseq的情况。
