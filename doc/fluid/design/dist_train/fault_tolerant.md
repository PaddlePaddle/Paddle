# Fluid Fault-Tolerant Training

## Background

We have implemented Elastic Deep Learning feature in previous PaddlePaddle versions,
see https://github.com/PaddlePaddle/edl and [this design](../../v2/design/cluster_train/README.md).

The new version of PaddlePaddle: Fluid is more flexible, so there are some changes when we
implement these both Fault-Tolerant and EDL with Fluid, this design will introduce the new design
we will use to implement EDL with Fluid. To be short, the new design only need the following
components:

1. Use etcd cluster to store process liveness and a queue to dispatch training datas.
1. Operators that do liveness report and watching.
1. Operators that can do transactional etcd enqueue/dequeue.
1. Fluid executors can run a block as a daemon thread for the liveness checks.


## Full Fault-Tolerant Training with Fluid

The default Fault-Tolerant using checkpoint feature have some limitations:

1. Processes on all nodes must be restarted and load the checkpoint from storage.
1. The offset of data reader is not saved, recovered job must train from start.
1. Can not add/delete trainer nodes during training, can not scale training resources.

With these limitations, we can not run a "big" job like to train and evaluate for weeks
without manual maintainous, because of the hardware failure.

To be "Full Fault-Tolerant", we will enable the distributed training job to be able to
detect hardware failures and recover the training process in a short time. To achieve
this, the following states must be recorded and watched by the job nodes, we 

- parameter server liveness
- trainer liveness
- distributed job queues recording the training data offsets

### Parameter Server Recovery

When one of the pserver goes down and then restarted by Kubernetes,
it will start on a different pod with a different network identity (IP address). Meanwhile,
trainers may still trying to send gradients to that non-existing server. So trainers must
watch the pserver liveness states and change the retry request target to the new recovered
server endpoint.

For sync training, when one of the parameter server recovers, it does not know the barrier
status when it fails. For example, if it fails when all pservers are in the state of
receiving "get" calls, then the recovered pserver will start to wait "send" calls, this may
cause the job wait for ever. 

We design the pserver can start with a "recovery mode", when it's automatically bringed up 
by Kubernetes, it should go into a "recovery mode", which will wait a specific timeout on the
current barrier. When training continues, the "recovery mode" is turned off automatically.
In general, pserver can start up with an option `--recovory` which enables the barrier condition
wait method for only one loop.

For async training, when one pserver is gone, the trainer simply skip the pserver updates when
the timeout reached.

### Trainer Recovery

Trainers will use etcd transactions to fetch training data chunks from "Todo" queue, and put to a
"Pending" queue.when one chunk is finished the chunk's index will be pushed to the etcd "Complete"
queue.

When one trainer fails, the data chunk should be in "Pending" queue, this chunk will timeout
and be pushed back to "Todo" later on. When the failed trainer is brought up by Kubernetes,
it will ask for a new chunk from "Todo" queue and continues the training.

Each trainer have a daemonized thread periatically obtain a distributed etcd lock and try finding
the timeout chunks and push them back to "Todo" if there are any.

For sync training, when one trainer fails, other trainers will wait until the failed trainer goes
up again and report it's barrier.

For async training, trainers can go up and down at any time. In this mode, increasing and decreasing
trainers are possible.

## Implementation

1. `listen_and_serv` op support "recovery mode" which can timeout the barrier waiting for the 
   first loop.
1. operators to manipulate etcd distributed queue including:
   - etcd_get_chunk: run at the beginning of every mini-batch, to retrieve training data.
   - etcd_enqueue_todo: run at the beginning of every pass, to add data for every pass.
   - etcd_enqueue_timeout: run at the end of every mini-batch, enqueue timeout chunk in
     "Pending" to "Todo".
   - etcd_state_watch: write server alive state, and watch corresponding etcd keys.
1. transpiler: add option to enable full-fault-tolerant, pass etcd endpoints to transpiler,
   transpiled pserver program and trainer program will use 
