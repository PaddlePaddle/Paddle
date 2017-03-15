# Distributed Training Design Doc

## Objective

We want Paddle to support training on a general-purpose cluster. The cluster runs Paddle, the Web server (e.g., Nginx), the log collector (e.g., fluentd), the distributed queue service (e.g., Kafka), the log joiner and other data processors written using Storm, Spark, and Hadoop MapReduce on the same cluster. As illustrated in the following graph:

![general purpose cluster](src/arch.png)

This poses new challenges for Paddle,

- Paddle need to be tault tolerant.
- Input training data can be online data from realtime logs, or batched data from distributed file system.
- User needs a simple way to train model on cloud. Complexities such as job scheduling should be hidden from user.

## Training Job

A training job will be created once user asks Paddle cloud to train a model. The training job is made up of different processes that collabratively consume data input and produce a trained model. There are three kind of processes:

- Master process
- Trainer process
- Parameter server process

One training job will only have one master process, typicall multiple trainer and parameter server processes. Their relation is illustrated in the following graph:

![process collabration](src/paddle-on-kubernetes-invited-blog-model-sharding.png)

### Master Process

Master process will:

- keep a list of alive trainers and a list of alive parameter servers and do *health check*,
  - if trainer is dead it will update task queue accordingly as mentioned in [task queue](#task-queue).
  - if a parameter server is dead or a new parameter server joins, it will broacast this information to all trainers.
- dispatches tasks to trainers. A *task* is a unit of data that a trainer needs to train on, and
- keep track of training progress on the dataset with *task queue*. Typically training will iterate on the dataset for a full pass until it goes into next pass.

#### Task Queue

Master process have three task queues to track training progress as shown in the graph below:

![task queues](src/paddle-task-queues.png)

- Todo queue holds tasks to be dispatched.
- Pending queue holds tasks that are currently training by trainers, and a mapping from trainers to their training tasks.
- Done queue holds tasks that are already trained.

A dataset will be sharded into tasks and dispatched by the master process. The life cycle of a single task is illustrated below:

![task states](src/paddle-task-states.png)

1. When a new pass of training starts, all tasks will be placed in the todo queue.
1. The master process will dispatch few tasks to each trainer at a time, puts them in pending queue and waits for completion.
1. The trainer will work on it's tasks and tell master once a task is completed. The master process will dispatch a new task to that trainer.
1. If a trainer is dead. the master process will move it's tasks back to the todo queue.
1. The master will move completed task to the done queue. When todo queue is empty, master will start a new pass by moving all tasks in done queue to todo queue.

### Trainer Process

Trainer process will train it's current tasks, tell parameter servers it's accumulated gradient, and download latest model from parameter servers.

Trainer holds entire network model while each parameter server hold a shard of model. So trainer needs to communicate will all parameter servers.

Communication involves two parts:

- upload accumulated gradient. Upload can be configured to happen every **n** mini-batches.
- download new model. Download can be configured to happend every **m** mini-batches. **n** and **m** does not need to be equal.

### Parameter Server Process

Parameter server processes hold model together. Since model parameters are sharded and saved on different parameter servers. All parameter servers collabratively form the global view of trained model.

## Fault Tolerant

TODO

## User Interface

TODO
