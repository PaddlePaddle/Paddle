# Checkpoint User Guide

## Background
In many cases, Stand-alone training and Distributed training can be aborted by the software problem or hardware problem. More seriously, we waste so much time and the performance of the machine but get nothing, which makes us frustrating and we have to restart it again.

## Purpose
The feature of ```Checkpoint``` can save Intermediate model variables, lookup table variable, and other needs data in checkpoint directory. When the exception occurs, we can load these variables from the checkpoint directory immediately.
## Introduce
### Complete Features Currently：
1. The Trainer 0 will save model variables in training.
2. Each of the Trainer will save its own arguments needed.
3. Each of the Parameter Server will save ```Distribute Lookup Table``` variables in training.
### Fluid Checkpoint directory structure：

```
checkpoint_dir (the checkpoint directory user define)
├── checkpoint_0 (the first save directory)
│   ├── __lockup_table__ (Distribute Lookup Table directory)
│   │   ├── table_pserver_0 (Lookup table's data about Pserver 0)
│   │   └── table_pserver_1
│   ├── __model__ (model directory)
│   │   └── var.w_1
│   └── trainer_0 (each trainer will save its own data)
│       ├── epoch_id
│       └── step_id
└── checkpoint_1 (the second save directory)
```

## usage
### Fluid.CheckpointConfig construct
When the user wants to use ```Checkpoint``` feature, the main thing user have to do is declare ```CheckpointConfig``` and construct it.

```CheckpointConfig``` has 4 member variables need to be initialized：

| Member Variable | Type | Comment | 
| - | :-: | - | 
| checkpoint_dir | int| checkpoint directory | 
| max_num_checkpoints | int | Maximum number of checkpoint copies | 
| epoch_interval | int |  epoch interval times |
| step_interval | int | step interval times |

### Add Fluid.CheckpointConfig's declaration in Fluid.Trainer
Because the initialization of Trainer needs an instance of ```CheckpointConfig```., we should declare ```CheckpointConfig``` in ```Fluid``` first.

For example：
```python
config = CheckpointConfig(
    checkpoint_dir = "/tmp/ckpt", max_num_checkpoints = 2, 
    epoch_interval = 2, step_interval = 10)
trainer = Trainer(..., checkpoint_config=config)
```

After all the things done, the train will save checkpoint at the specified epoch and step, when the train is aborted, the user can restart it, the train will restore from the latest copy.

## Related API
[Related Trainer API](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/trainer.py)

## Attention
1. Make the ```checkpoint_dir``` only be used by one train job.
2. The number of ```max_num_checkpoints``` need to be adjusted by the disk size and model size.
3. Too frequently to slow down the train speed, so too ```small epoch_interval``` and ```step_interval``` are not suitable.
4. **In distributed train**, each Trainer will save arguments in its ```checkpoint_dir``` (Only Trainer 0 will save model variables). We need **distributed file system (HDFS, etc)** to merge all the ```checkpoint_dir``` to get the whole data.
