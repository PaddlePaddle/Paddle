# Checkpoint功能使用指南

## 背景
单机/多机在训练过程中会由于软件/硬件的问题出现异常，导致训练中断，进而导致训练无结果或结果不可用，浪费大量时间和机器性能。

## 目的
Checkpoint功能能够在训练中途对训练数据中间数据进行保存，出现异常恢复训练的时候能够加载中途保存的数据继续训练， 实现单机/多机的容错训练的功能。

## 说明
### 目前已实现的参数保存：
1. 基于Trainer 0 实现训练过程中的参数保存
2. 基于PServer 实现了```Distribute Lookup Table```相关参数保存
### Fluid Checkpoint 保存数据目录结构：

```
checkpoint_dir (用户定义的checkpoint目录)
├── checkpoint_0 (第一次保存)
│   ├── __lockup_table__ (Distribute Lookup Table 目录)
│   │   ├── table_pserver_0 (Pserver 0 号保存的lookup table 数据)
│   │   └── table_pserver_1
│   ├── __model__ (model 目录)
│   │   └── var.w_1
│   └── trainer_0 (trainer 自有数据保存)
│       ├── epoch_id
│       └── step_id
└── checkpoint_1 (第二次保存)
```

## 使用方法
### 声明Fluid.CheckpointConfig
用户对checkpoint功能的配置，主要是配置对象```Fluid```中的```CheckpointConfig```.

```CheckpointConfig``` 包括4个参数：

| 参数 | 类型 | 说明 | 
| - | :-: | - | 
| checkpoint_dir | int| checkpoint存储目录 | 
| max_num_checkpoints | int | 最大保存的checkpoint副本数 | 
| epoch_interval | int | 每隔epoch_interval轮epoch |
| step_interval | int | 每隔step_interval轮step |

### 在Fluid.Trainer对象的声明中加入Fluid.CheckpointConfig的声明
Trainer的__init__方法的参数中包含了对```CheckpointConfig```， 需要传入在声明Trainer前声明的```CheckpointConfig```对象。
如：
```python
config = CheckpointConfig(
    checkpoint_dir = "/tmp/ckpt", max_num_checkpoints = 2, 
    epoch_interval = 2, step_interval = 10)
trainer = Trainer(..., checkpoint_config=config)
```
定义和声明完成后， 训练在运行过程中就会在指定的step和epoch处进行保存，出现异常时，就会自动从最新的checkpoint目录进行参数恢复啦！

## 相关API
[Trainer API 说明](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/trainer.py)

## 注意
1. 保证每个训练的```checkpoint_dir``` 与其他训练独立。
2. 最大副本数量```max_num_checkpoints```需要根据磁盘容量以及模型的大小进行调整， 保证磁盘的可用性。
3. ```epoch_interval```  和 ```step_interval```  不宜过小， 频繁的进行checkpoint会拖慢训练速度。
4. **分布式训练**的过程中：每个Trainer都会在```checkpoint_dir```目录中保存当前Trainer的参数（只有Trainer 0会保存模型的参数），需要**分布式文件系统(HDFS等)**将同```checkpoint_dir```目录的数据进行合并才能得到完整的数据，恢复训练的时候需要用完整的数据进行恢复。
