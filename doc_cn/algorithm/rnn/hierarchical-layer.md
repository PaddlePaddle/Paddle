# 支持双层RNN的Layer介绍

## pooling_layer

pooling_layer的使用示例如下，详细见配置API。
```python
seq_pool = pooling_layer(input=layer,
                         pooling_type=AvgPooling(),
                         agg_level=AggregateLevel.EACH_SEQUENCE)
```
- `pooling_type`有两种，分别是MaxPooling()和AvgPooling。
- `agg_level=AggregateLevel.TIMESTEP`时（默认值）：
  - 作用：双层变0层，或单层变0层
  - 输入：双层seq或单层seq
  - 输出：0层seq（一个向量），即整个seq的平均值（或最大值）
- `agg_level=AggregateLevel.EACH_SEQUENCE`时：
  - 作用：双层变单层
  - 输入：必须是双层seq
  - 输出：单层seq，其中每个向量是原来双层seq中每个subseq的平均值（或最大值）

## last_seq和first_seq

last_seq的使用示例如下（first_seq类似），详细见配置API。
```python
last = last_seq(input=layer,
                agg_level=AggregateLevel.EACH_SEQUENCE)
```
- `agg_level=AggregateLevel.TIMESTEP`时（默认值）：
  - 作用：双层变0层，或单层变0层
  - 输入：双层seq或单层seq
  - 输出：0层seq（一个向量），即整个seq最后（或最开始）的一个向量。
- `agg_level=AggregateLevel.EACH_SEQUENCE`时：
  - 作用：双层变单层
  - 输入：必须是双层seq
  - 输出：单层seq，其中每个向量是原来双层seq中每个subseq最后（或最开始）的一个向量。

## expand_layer

expand_layer的使用示例如下，详细见配置API。
```python
expand = expand_layer(input=layer1,
                      expand_as=layer2,
                      expand_level=ExpandLevel.FROM_TIMESTEP)
```
- `expand_level=ExpandLevel.FROM_TIMESTEP`时（默认值）：
  - 作用：0层变双层，或单层变双层
  - 输入：layer1必须是0层seq，layer2可以是双层seq或单层seq
  - 输出：单或双层seq（和layer2的一样），其中第i个seq中每个向量的值均为layer1中第i个向量的值
- `expand_level=ExpandLevel.FROM_SEQUENCE`时：
  - 作用：单层变双层
  - 输入：layer1必须是单层seq，layer2必须是双层seq
  - 输出：双层seq（和layer2的一样），其中第i个subseq中每个向量的值均为layer1中第i个向量的值 