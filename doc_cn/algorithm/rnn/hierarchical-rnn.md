# 双层RNN配置与示例

我们在`paddle/gserver/tests/test_RecurrentGradientMachine`单测中，通过多组语义相同的单双层RNN配置，讲解如何使用双层RNN。

## 示例1：双进双出，subseq间无memory

配置：单层RNN（`sequence_layer_group`）和双层RNN（`sequence_nest_layer_group`），语义完全相同。

### 读取双层序列的方法

首先，我们看一下单双层序列的不同数据组织形式（您也可以采用别的组织形式）：

- 单层序列的数据（`Sequence/tour_train_wdseg`）如下，一共有10个样本。每个样本由两部分组成，一个label（此处都为2）和一个已经分词后的句子。

```text
2  	酒店 有 很 舒适 的 床垫 子 ， 床上用品 也 应该 是 一人 一 换 ， 感觉 很 利落 对 卫生 很 放心 呀 。
2  	很 温馨 ， 也 挺 干净 的 * 地段 不错 ， 出来 就 有 全家 ， 离 地铁站 也 近 ， 交通 很方便 * 就是 都 不 给 刷牙 的 杯子 啊 ， 就 第一天 给 了 一次性杯子 *
2  	位置 方便 ， 强烈推荐 ， 十一 出去玩 的 时候 选 的 ， 对面 就是 华润万家 ， 周围 吃饭 的 也 不少 。
2  	交通便利 ， 吃 很 便利 ， 乾 浄 、 安静 ， 商务 房 有 电脑 、 上网 快 ， 价格 可以 ， 就 早餐 不 好吃 。 整体 是 不错 的 。 適 合 出差 來 住 。
2  	本来 准备 住 两 晚 ， 第 2 天 一早 居然 停电 ， 且 无 通知 ， 只有 口头 道歉 。 总体来说 性价比 尚可 ， 房间 较 新 ， 还是 推荐 .
2  	这个 酒店 去过 很多 次 了 ， 选择 的 主要原因 是 离 客户 最 便宜 相对 又 近 的 酒店
2  	挺好 的 汉庭 ， 前台 服务 很 热情 ， 卫生 很 整洁 ， 房间 安静 ， 水温 适中 ， 挺好 ！
2  	HowardJohnson 的 品质 ， 服务 相当 好 的 一 家 五星级 。 房间 不错 、 泳池 不错 、 楼层 安排 很 合理 。 还有 就是 地理位置 ， 简直 一 流 。 就 在 天一阁 、 月湖 旁边 ， 离 天一广场 也 不远 。 下次 来 宁波 还会 住 。
2  	酒店 很干净 ， 很安静 ， 很 温馨 ， 服务员 服务 好 ， 各方面 都 不错 *
2  	挺好 的 ， 就是 没 窗户 ， 不过 对 得 起 这 价格
```

- 双层序列的数据（`Sequence/tour_train_wdseg.nest`）如下，一共有4个样本。样本间用空行分开，代表不同的双层序列，序列数据和上面的完全一样。每个样本的子句数分别为2,3,2,3。

```text
2  	酒店 有 很 舒适 的 床垫 子 ， 床上用品 也 应该 是 一人 一 换 ， 感觉 很 利落 对 卫生 很 放心 呀 。
2  	很 温馨 ， 也 挺 干净 的 * 地段 不错 ， 出来 就 有 全家 ， 离 地铁站 也 近 ， 交通 很方便 * 就是 都 不 给 刷牙 的 杯子 啊 ， 就 第一天 给 了 一次性杯子 *

2  	位置 方便 ， 强烈推荐 ， 十一 出去玩 的 时候 选 的 ， 对面 就是 华润万家 ， 周围 吃饭 的 也 不少 。
2  	交通便利 ， 吃 很 便利 ， 乾 浄 、 安静 ， 商务 房 有 电脑 、 上网 快 ， 价格 可以 ， 就 早餐 不 好吃 。 整体 是 不错 的 。 適 合 出差 來 住 。
2  	本来 准备 住 两 晚 ， 第 2 天 一早 居然 停电 ， 且 无 通知 ， 只有 口头 道歉 。 总体来说 性价比 尚可 ， 房间 较 新 ， 还是 推荐 .

2  	这个 酒店 去过 很多 次 了 ， 选择 的 主要原因 是 离 客户 最 便宜 相对 又 近 的 酒店
2  	挺好 的 汉庭 ， 前台 服务 很 热情 ， 卫生 很 整洁 ， 房间 安静 ， 水温 适中 ， 挺好 ！

2  	HowardJohnson 的 品质 ， 服务 相当 好 的 一 家 五星级 。 房间 不错 、 泳池 不错 、 楼层 安排 很 合理 。 还有 就是 地理位置 ， 简直 一 流 。 就 在 天一阁 、 月湖 旁边 ， 离 天一广场 也 不远 。 下次 来 宁波 还会 住 。
2  	酒店 很干净 ， 很安静 ， 很 温馨 ， 服务员 服务 好 ， 各方面 都 不错 *
2  	挺好 的 ， 就是 没 窗户 ， 不过 对 得 起 这 价格
```

其次，我们看一下单双层序列的不同dataprovider（见`sequenceGen.py`）：

- 单层序列的dataprovider如下：
  - word_slot是integer_value_sequence类型，代表单层序列。
  - label是integer_value类型，代表一个向量。

```python
def hook(settings, dict_file, **kwargs):
    settings.word_dict = dict_file
    settings.input_types = [integer_value_sequence(len(settings.word_dict)), 
                            integer_value(3)]

@provider(init_hook=hook)
def process(settings, file_name):
    with open(file_name, 'r') as fdata:
        for line in fdata:
            label, comment = line.strip().split('\t')
            label = int(''.join(label.split()))
            words = comment.split()
            word_slot = [settings.word_dict[w] for w in words if w in settings.word_dict]
            yield word_slot, label
```

- 双层序列的dataprovider如下：
  - word_slot是integer_value_sub_sequence类型，代表双层序列。
  - label是integer_value_sequence类型，代表单层序列，即一个子句一个label。注意：也可以为integer_value类型，代表一个向量，即一个句子一个label。通常根据任务需求进行不同设置。
  - 关于dataprovider中input_types的详细用法，参见PyDataProvider2。

```python
def hook2(settings, dict_file, **kwargs):
    settings.word_dict = dict_file
    settings.input_types = [integer_value_sub_sequence(len(settings.word_dict)),
                            integer_value_sequence(3)]

@provider(init_hook=hook2)
def process2(settings, file_name):
    with open(file_name) as fdata:
        label_list = []
        word_slot_list = []
        for line in fdata:
            if (len(line)) > 1:
                label,comment = line.strip().split('\t')
                label = int(''.join(label.split()))
                words = comment.split()
                word_slot = [settings.word_dict[w] for w in words if w in settings.word_dict]
                label_list.append(label)
                word_slot_list.append(word_slot)
            else:
                yield word_slot_list, label_list
                label_list = []
                word_slot_list = []
```

### 模型中的配置

首先，我们看一下单层序列的配置（见`sequence_layer_group.conf`）。注意：batchsize=5表示一次过5句单层序列，因此2个batch就可以完成1个pass。

```python
settings(batch_size=5)

data = data_layer(name="word", size=dict_dim)

emb = embedding_layer(input=data, size=word_dim)

# (lstm_input + lstm) is equal to lstmemory 
with mixed_layer(size=hidden_dim*4) as lstm_input:
    lstm_input += full_matrix_projection(input=emb)

lstm = lstmemory_group(input=lstm_input,
                       size=hidden_dim,
                       act=TanhActivation(),
                       gate_act=SigmoidActivation(),
                       state_act=TanhActivation(),
                       lstm_layer_attr=ExtraLayerAttribute(error_clipping_threshold=50))

lstm_last = last_seq(input=lstm)

with mixed_layer(size=label_dim, 
                 act=SoftmaxActivation(), 
                 bias_attr=True) as output:
    output += full_matrix_projection(input=lstm_last)

outputs(classification_cost(input=output, label=data_layer(name="label", size=1)))

```
其次，我们看一下语义相同的双层序列配置（见`sequence_nest_layer_group.conf`），并对其详细分析：

- batchsize=2表示一次过2句双层序列。但从上面的数据格式可知，2句双层序列和5句单层序列的数据完全一样。
- data_layer和embedding_layer不关心数据是否是序列格式，因此两个配置在这两层上的输出是一样的。
- lstmemory:
  - 单层序列过了一个mixed_layer和lstmemory_group。
  - 双层序列在同样的mixed_layer和lstmemory_group外，直接加了一层group。由于这个外层group里面没有memory，表示subseq间不存在联系，即起到的作用仅仅是把双层seq拆成单层，因此双层序列过完lstmemory的输出和单层的一样。
- last_seq：
  - 单层序列直接取了最后一个元素
  - 双层序列首先（last_seq层）取了每个subseq的最后一个元素，将其拼接成一个新的单层序列；接着（expand_layer层）将其扩展成一个新的双层序列，其中第i个subseq中的所有向量均为输入的单层序列中的第i个向量；最后（average_layer层）取了每个subseq的平均值。
  - 分析得出：第一个last_seq后，每个subseq的最后一个元素就等于单层序列的最后一个元素，而expand_layer和average_layer后，依然保持每个subseq最后一个元素的值不变（这两层仅是为了展示它们的用法，实际中并不需要）。因此单双层序列的输出是一样旳。

```python
settings(batch_size=2)

data = data_layer(name="word", size=dict_dim)

emb_group = embedding_layer(input=data, size=word_dim)

# (lstm_input + lstm) is equal to lstmemory 
def lstm_group(lstm_group_input):
    with mixed_layer(size=hidden_dim*4) as group_input:
      group_input += full_matrix_projection(input=lstm_group_input)

    lstm_output = lstmemory_group(input=group_input,
                                  name="lstm_group",
                                  size=hidden_dim,
                                  act=TanhActivation(),
                                  gate_act=SigmoidActivation(),
                                  state_act=TanhActivation(),
                                  lstm_layer_attr=ExtraLayerAttribute(error_clipping_threshold=50))
    return lstm_output

lstm_nest_group = recurrent_group(input=SubsequenceInput(emb_group),
                                  step=lstm_group,
                                  name="lstm_nest_group")
# hasSubseq ->(seqlastins) seq
lstm_last = last_seq(input=lstm_nest_group, agg_level=AggregateLevel.EACH_SEQUENCE)

# seq ->(expand) hasSubseq
lstm_expand = expand_layer(input=lstm_last, expand_as=emb_group, expand_level=ExpandLevel.FROM_SEQUENCE)

# hasSubseq ->(average) seq
lstm_average = pooling_layer(input=lstm_expand,
                             pooling_type=AvgPooling(),
                             agg_level=AggregateLevel.EACH_SEQUENCE)

with mixed_layer(size=label_dim, 
                 act=SoftmaxActivation(), 
                 bias_attr=True) as output:
    output += full_matrix_projection(input=lstm_average)

outputs(classification_cost(input=output, label=data_layer(name="label", size=1)))
```
## 示例2：双进双出，subseq间有memory

配置：单层RNN（`sequence_rnn.conf`），双层RNN（`sequence_nest_rnn.conf`和`sequence_nest_rnn_readonly_memory.conf`），语义完全相同。

### 读取双层序列的方法

我们看一下单双层序列的不同数据组织形式和dataprovider（见`rnn_data_provider.py`）
```python
data = [
    [[[1, 3, 2], [4, 5, 2]], 0],
    [[[0, 2], [2, 5], [0, 1, 2]], 1],
]

@provider(input_types=[integer_value_sub_sequence(10),
                       integer_value(3)])
def process_subseq(settings, file_name):
    for d in data:
        yield d

@provider(input_types=[integer_value_sequence(10),
                       integer_value(3)])
def process_seq(settings, file_name):
    for d in data:
        seq = []
```
- 单层序列：有两句，分别为[1,3,2,4,5,2]和[0,2,2,5,0,1,2]。
- 双层序列：有两句，分别为[[1,3,2],[4,5,2]]（2个子句）和[[0,2],[2,5],[0,1,2]]（3个子句）。
- 单双层序列的label都分别是0和1

### 模型中的配置

我们选取单双层序列配置中的不同部分，来对比分析两者语义相同的原因。

- 单层序列：过了一个很简单的recurrent_group。每一个时间步，当前的输入y和上一个时间步的输出rnn_state做了一个全链接。

```python
def step(y):
    mem = memory(name="rnn_state", size=hidden_dim)
    return fc_layer(input=[y, mem],
                    size=hidden_dim,
                    act=TanhActivation(),
                    bias_attr=True,
                    name="rnn_state")

out = recurrent_group(step=step, input=emb)
```
- 双层序列，外层memory是一个元素：
  - 内层inner_step的recurrent_group和单层序列的几乎一样。除了boot_layer=outer_mem，表示将外层的outer_mem作为内层memory的初始状态。外层outer_step中，outer_mem是一个子句的最后一个向量，即整个双层group是将前一个子句的最后一个向量，作为下一个子句memory的初始状态。
  - 从输入数据上看，单双层序列的句子是一样的，只是双层序列将其又做了子序列划分。因此双层序列的配置中，必须将前一个子句的最后一个元素，作为boot_layer传给下一个子句的memory，才能保证和单层序列的配置中“每一个时间步都用了上一个时间步的输出结果”一致。

```python
def outer_step(x):
    outer_mem = memory(name="outer_rnn_state", size=hidden_dim)
    def inner_step(y):
        inner_mem = memory(name="inner_rnn_state",
                           size=hidden_dim,
                           boot_layer=outer_mem)
        return fc_layer(input=[y, inner_mem],
                        size=hidden_dim,
                        act=TanhActivation(),
                        bias_attr=True,
                        name="inner_rnn_state")

    inner_rnn_output = recurrent_group(
        step=inner_step,
        input=x)
    last = last_seq(input=inner_rnn_output, name="outer_rnn_state")

    return inner_rnn_output

out = recurrent_group(step=outer_step, input=SubsequenceInput(emb))
```
- 双层序列，外层memory是单层序列：
  - 由于外层每个时间步返回的是一个子句，这些子句的长度往往不等长。因此当外层有is_seq=True的memory时，内层是**无法直接使用**它的，即内层memory的boot_layer不能链接外层的这个memory。
  - 如果内层memory想**间接使用**这个外层memory，只能通过`pooling_layer`、`last_seq`或`first_seq`这三个layer将它先变成一个元素。但这种情况下，外层memory必须有boot_layer，否则在第0个时间步时，由于外层memory没有任何seq信息，因此上述三个layer的前向会报出“**Check failed: input.sequenceStartPositions**”的错误。

## 示例3：双进双出，输入不等长

**输入不等长**是指recurrent_group的多个输入在各时刻的长度可以不相等, 但需要指定一个和输出长度一致的input，用<font color="red">targetInlink</font>表示。参考配置：单层RNN（`sequence_rnn_multi_unequalength_inputs.conf`），双层RNN（`sequence_nest_rnn_multi_unequalength_inputs.conf`）

### 读取双层序列的方法

我们看一下单双层序列的数据组织形式和dataprovider（见`rnn_data_provider.py`）
```python
data2 = [
    [[[1, 2], [4, 5, 2]], [[5, 4, 1], [3, 1]] ,0],
    [[[0, 2], [2, 5], [0, 1, 2]],[[1, 5], [4], [2, 3, 6, 1]], 1],
]

@provider(input_types=[integer_value_sub_sequence(10),
                       integer_value_sub_sequence(10),
                       integer_value(2)],
          should_shuffle=False)
def process_unequalength_subseq(settings, file_name): #双层RNN的dataprovider
    for d in data2:
        yield d


@provider(input_types=[integer_value_sequence(10),
                       integer_value_sequence(10),
                       integer_value(2)],
          should_shuffle=False)
def process_unequalength_seq(settings, file_name): #单层RNN的dataprovider
    for d in data2:
        words1=reduce(lambda x,y: x+y, d[0])
        words2=reduce(lambda x,y: x+y, d[1])
        yield words1, words2, d[2]
```

data2 中有两个样本，每个样本有两个特征, 记fea1, fea2。

- 单层序列：两个样本分别为[[1, 2, 4, 5, 2], [5, 4, 1, 3, 1]] 和 [[0, 2, 2, 5, 0, 1, 2], [1, 5, 4, 2, 3, 6, 1]]
- 双层序列：两个样本分别为
  - **样本1**：[[[1, 2], [4, 5, 2]], [[5, 4, 1], [3, 1]]]。fea1和fea2都分别有2个子句，fea1=[[1, 2], [4, 5, 2]], fea2=[[5, 4, 1], [3, 1]]
  - **样本2**：[[[0, 2], [2, 5], [0, 1, 2]],[[1, 5], [4], [2, 3, 6, 1]]]。fea1和fea2都分别有3个子句， fea1=[[0, 2], [2, 5], [0, 1, 2]], fea2=[[1, 5], [4], [2, 3, 6, 1]]。<br/>
  - **注意**：每个样本中，各特征的子句数目需要相等。这里说的“双进双出，输入不等长”是指fea1在i时刻的输入的长度可以不等于fea2在i时刻的输入的长度。如对于第1个样本，时刻i=2, fea1[2]=[4, 5, 2]，fea2[2]=[3, 1]，3≠2。
- 单双层序列中，两个样本的label都分别是0和1

### 模型中的配置

单层RNN（`sequence_rnn_multi_unequalength_inputs.conf`）和双层RNN（`sequence_nest_rnn_multi_unequalength_inputs.conf`）两个模型配置达到的效果完全一样，区别只在于输入为单层还是双层序列，现在我们来看它们内部分别是如何实现的。

- 单层序列：
  - 过了一个简单的recurrent_group。每一个时间步，当前的输入y和上一个时间步的输出rnn_state做了一个全连接，功能与示例2中`sequence_rnn.conf`的`step`函数完全相同。这里，两个输入x1,x2分别通过calrnn返回最后时刻的状态。结果得到的encoder1_rep和encoder2_rep分别是单层序列，最后取encoder1_rep的最后一个时刻和encoder2_rep的所有时刻分别相加得到context。
  - 注意到这里recurrent_group输入的每个样本中，fea1和fea2的长度都分别相等，这并非偶然，而是因为recurrent_group要求输入为单层序列时，所有输入的长度都必须相等。

```python
def step(x1, x2):
	def calrnn(y):
		mem = memory(name = 'rnn_state_' + y.name, size = hidden_dim)
        out = fc_layer(input = [y, mem],
	        size = hidden_dim,
	        act = TanhActivation(),
            bias_attr = True,
            name = 'rnn_state_' + y.name)
        return out

	encoder1 = calrnn(x1)
    encoder2 = calrnn(x2)
    return [encoder1, encoder2]
    
encoder1_rep, encoder2_rep = recurrent_group(
    name="stepout",                           
    step=step,
    input=[emb1, emb2])

encoder1_last = last_seq(input = encoder1_rep)                           
encoder1_expandlast = expand_layer(input = encoder1_last,
                                   expand_as = encoder2_rep)
context = mixed_layer(input = [identity_projection(encoder1_expandlast),
                               identity_projection(encoder2_rep)],
                      size = hidden_dim)
```
- 双层序列：
  - 双层RNN中，对输入的两个特征分别求时序上的连续全连接(`inner_step1`和`inner_step2`分别处理fea1和fea2)，其功能与示例2中`sequence_nest_rnn.conf`的`outer_step`函数完全相同。不同之处是，此时输入`[SubsequenceInput(emb1), SubsequenceInput(emb2)]`在各时刻并不等长。
  - 函数`outer_step`中可以分别处理这两个特征，但我们需要用<font color=red>targetInlink</font>指定recurrent_group的输出的格式（各子句长度）只能和其中一个保持一致，如这里选择了和emb2的长度一致。
  - 最后，依然是取encoder1_rep的最后一个时刻和encoder2_rep的所有时刻分别相加得到context。

```python
def outer_step(x1, x2):
    outer_mem1 = memory(name = "outer_rnn_state1", size = hidden_dim)
    outer_mem2 = memory(name = "outer_rnn_state2", size = hidden_dim)
    def inner_step1(y):
        inner_mem = memory(name = 'inner_rnn_state_' + y.name,
                           size = hidden_dim,
                           boot_layer = outer_mem1)
        out = fc_layer(input = [y, inner_mem],
                       size = hidden_dim,
                       act = TanhActivation(),
                       bias_attr = True,
                       name = 'inner_rnn_state_' + y.name)
        return out

    def inner_step2(y):
        inner_mem = memory(name = 'inner_rnn_state_' + y.name,
                           size = hidden_dim,
                           boot_layer = outer_mem2)
        out = fc_layer(input = [y, inner_mem],
                       size = hidden_dim,
                       act = TanhActivation(),
                       bias_attr = True,
                       name = 'inner_rnn_state_' + y.name)
        return out

    encoder1 = recurrent_group(
        step = inner_step1,
        name = 'inner1',
        input = x1)

    encoder2 = recurrent_group(
        step = inner_step2,
        name = 'inner2',
        input = x2)

    sentence_last_state1 = last_seq(input = encoder1, name = 'outer_rnn_state1')
    sentence_last_state2_ = last_seq(input = encoder2, name = 'outer_rnn_state2')

    encoder1_expand = expand_layer(input = sentence_last_state1,
                                   expand_as = encoder2)

    return [encoder1_expand, encoder2]

encoder1_rep, encoder2_rep = recurrent_group(
    name="outer",
    step=outer_step,
    input=[SubsequenceInput(emb1), SubsequenceInput(emb2)],
    targetInlink=emb2)

encoder1_last = last_seq(input = encoder1_rep)
encoder1_expandlast = expand_layer(input = encoder1_last,
                                   expand_as = encoder2_rep)
context = mixed_layer(input = [identity_projection(encoder1_expandlast),
                               identity_projection(encoder2_rep)],
                      size = hidden_dim)
```

## 示例4：beam_search的生成

TBD