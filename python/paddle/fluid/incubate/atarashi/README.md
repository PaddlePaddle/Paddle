# ATARSHI for paddle
1. 分布式训练管理(Summary/ Metrics/ Tensorboard 以及其他你喜欢的高级轮子)
2. data pipeline (依靠`paddle pyreader`, 提供灵活的Dataset API供用户组件自定义数据流)
3. 数据管理（tokenization, 二进制化...)

## Getting Started|快速开始
依赖:

1. paddle 1.3.0 
2. tensorboardX

用法
```python

    #模型五部曲
    class Model(object):
        def __init__(self, config, mode):
            self.embedding = Embedding(config['emb_size'], config['vocab_size'])
            self.fc1 = FC(config['hidden_size'])
            self.fc2 = FC(config['hidden_size']

        def forward(self, features):
            q, t = features 
            q_emb = softsign(self.embedding(q))
            t_emb = softsign(self.embedding(t))
            atarashi.summary.histogram('query_embedding', q_emb) #记录tensorboard
            q_emb = self.fc1(q_emb)
            t_emb = self.fc2(t_emn)
            prediction = dot(q_emb,  emb)
            return prediction

        def loss(self, predictions, label):
            return sigmoid_cross_entropy_with_logits(predictions, label)

        def backward(self, loss):
            opt = AdamOptimizer(1.e-3)
            opt.mimize(loss)

        def metrics(self, predictions, label):
            auc = atarshi.metrics.Auc(predictions, label)
            return {'auc': auc}

    # 超参可以来自于文件/ 环境变量/ 命令行
    run_config = atarashi.parse_runconfig(args)
    hparams = atarashi.parse_hparam(args)
    
    #`FeatureColumns` 用于管理训练、预测文件. 会自动进行二进制化.
    feature_column = atarashi.data.FeatureColumns(columns=[
            atarashi.data.TextColumn('query', vocab='./vocab'),
            atarashi.data.TextColumn('title', vocab='./vocab'),
            atarashi.data.LabelColumn('label'),
        ])

    # 生成训练、预测集.
    train_ds = feature_column.build_dataset(data_dir='./data',  shuffle=True, repeat=True)
    eval_ds = feature_column.build_dataset(data_dir='./data', shuffle=False, repeat=False)

    # 边Train 边Eval 并且输出Tensorboard可视化
    atarashi.train_and_eval(Model, hparams, run_config, train_ds, eval_ds)
```
详细见example/toy/

## 主要构件
1. train_and_eval

    根据`--hparam`指定的超参分别初始化*训练*模型以及*预测*模型.
    根据`--run_config` 指定训练参数.
    开始训练，同时执行预测。结果以tensorboard的形式呈现

2. FeatureColumns
    
    用`FeatureColumns`来管理训练数据. 根据自定义`Column`来适配多种ML任务（NLP/CV...).
    `FeatureColumns`会自动对提供的训练数据进行批量预处理(tokenization, 查词表, etc.)并二进制化，并且生成训练用的dataset

3. Dataset

    如果觉得`FeatureColumns`太局限。可以使用`atarashi.Dataset.from_generator`来构造自己的dataset，配合你最熟悉的shuffle/ interleave/ padded_batch/ repeat 满足定制化需求.
    p.s. Dataset的前处理中可以使用numpy，提高了paddle程序的灵活性.

4. MonitoredExecutor

    如果你还觉得`train_and_eval`太局限。可以使用`atarashi.MonitoredExecutor`来构造自的Train Loop! (其实`train_and_eval`不过是把两个`atarashi.MonitoredExecutor`套起来而已)

5. Summary

    像TF一样画tensorboard吧!

## Running the tests|测试
...

## Contributing|如何贡献

1. 欢迎贡献！
2. functional programing is welcomed

## Discussion|讨论
...


## TODO
1. dataset output_types/ output_shapes 自动推断
2. 自动超参数搜索
3. 分布式同步/ 异步自动化并行
4. ...
