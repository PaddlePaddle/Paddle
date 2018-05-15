# PaddlePaddle Design Doc

## Ingredients

As our design principle is starting from the essence: how could we
allow users to express and solve their problems as neural networks.
Some essential concepts that our API have to provide include:

1. A *topology* is an expression of *layers*.

1. A layer could be any kind of computation, including *cost*.

1. Some layers have parameters, some don't. Most costs don't have
   parameters.

1. In some topologies, layers share parameters.  For
   example,
   [the network for training a ranking model](https://github.com/PaddlePaddle/Paddle/issues/1311#issuecomment-279121850).

1. At programming time, users specify topologies and possible sharing
   of parameters.  PaddlePaddle can figure out and create parameters
   required (and possibly shared) by one or more topologies.


## Starting from Examples

As a summarization
of
[our disucssion](https://github.com/PaddlePaddle/Paddle/issues/1315),
let us present two examples here:


### Example 1. Sharing Parameters between Layers

We use
the
[3-branch ranking](https://github.com/PaddlePaddle/Paddle/issues/1311#issuecomment-279121850) model
in this example.  For your convenience, I copy-a-paste the model's
topology as follows:

```
A -> f -\
Q -> f --> cost
B -> f -/
```

The following program trains the topology including the cost, and then
use the sub-network in the trained topology in inference:

```python
def f(in):
    e = paddle.layer.embedding(in, parameter_name="embedding")
    o = paddle.layer.softmax(e, parameter_name="semantic")
    return o

# Create 3 topologies (subnets), they share parameters because all
# correspoinding layers have the same parameter names.
fA = f(paddle.layer.data(input_name="A"))
fB = f(paddle.layer.data(input_name="B"))
fQ = f(paddle.layer.data(input_name="Q"))

topology = paddle.layer.less_than(
               paddle.layer.cross_entropy(fA, fQ),
               paddle.layer.corss_entropy(fB, fQ))

# Derive parameters required in topology and create them in model.
parameters = paddle.parameters.create(topology)

# Estimate parameters used in topology from data.
paddle.train(topology, parameters, reader=read_ranking_model_data)

# Inference using fA (or fB or fC, as they share their parameters).
[testA, testB, testQ] = read_ranking_model_data()
print "The sematic-vector of testA: ", paddle.infer(fA, parameters, testA)
```


### Example 2. Sharing Parameters between "Models"

We use GAN in this example.  In the following example program, `d0` and `d1`
correspond to the two networks in the following figure:

<img src="https://github.com/wangyang59/book/raw/00036f4b0da5225041a6824587c1a01cf20159b1/gan/image/gan_ig.png" width=400 />

```python
def G(in):
    # over-simplified example as G has only one layers:
    return paddle.layer.fc(in, parameter_name="G")

def D(in);
    # again, over-simplified:
    return paddle.layer.fc(in, parameter_name="D")

# Construct the first topology, which contains both D and G.
# By learning this topology, we update parameters of G.
d0 = paddle.layer.should_be_false(D(G(paddle.layer.data())))

# Construct a second topology d1, which contains only D. By
# training this topology, we update parameters of D.  Note
# that d1 share parameters with d0.
d1 = paddle.layer.should_be_true(D(paddle.layer.data()))

# Create parameters from a list of multiple topologies (models) for
# the chance to share parameters between these topologies.
parameters = paddle.parameters.create([d0, d1])

# Iterative training of GAN.
for ...:
    train(d0, parameters, reader=read_from_rng, immutable_parameters={"D"})
    train(d1, parameters, reader=read_from_realistic_images)

# Use d1 for inference:
print "D thinks a batch of images are realistic ", infer(d1, parameters, read_mnist_images)
```


### Summarization


Above two programs reveal some important design concerns:

1. Users describe a topology as an expression of layers.  Every layer
   has a *parameter name*.  If the users don't specify it explicitly, it's automatically generated as a unique name.  By
   specifying the parameter name, users can specify the sharing of
   parameters between layers and even between topologies.

1. `paddle.parameters.create` figures out parameters required by one
   or more topologies from parameter names of layers.  It creates these
   parameters and returns a `ParameterSet` object, which is in essence
   a map from *parameter names* to *parameters*.

1. At training and inference time, `paddle.train` and `paddle.infer`
   requires both a topology and the parameter set that holds the parameters of that topology.  There are some reasons:

   1. This prevents users from forgetting to call
      `paddle.parameters.create`.
   1. `paddle.train` needs to know which parameter set to update.
   1. Users could load another (pre-trained) parameter set and use it
      with a topology in `train.infer`.

1. By specifying the `immutable_parameters` parameter of
   `paddle.train`, we can forbid the update of these parameters.


## Reader

Not all programming frameworks allow users to define I/O functions.
An example is Google MapReduce, which can only read from text,
SSTable, and RecordIO files.  Hadoop MapReduce allows users to define
readers and writers by deriving from base classes `Reader` and
`Writer`.  The former is less flexible but also less error-prone.  We
decide to provide the flexibility to users to define their readers.


There are some open questions here:

1. **Should a reader return a Python dictionary?**

1. **How to map multiple outputs from a reader to multiple data layers?**

1. **How to easily compose some existing readers to read more data and
   feed a topology with more data layers?**


## Training

The recommended way to training a model is to call `paddle.train`,
which simply calls `paddle.trainer.Default`, a global variable of
type `paddle.trainer.SGD`.  Equivalently, we can do

```python
opt = paddle.trainer.SGD(..., paddle.updater.Adam(...))
opt.train(topology, parameters, reader=read, ...)
```

### Updater

Please be aware that a trainer can accept an updater as its data
member, where an updater is a class derived from
`paddle.trainer.Updater`.  This is to make it easier to customize
trainers, as discussed
[here](https://github.com/PaddlePaddle/Paddle/issues/1319).

### Event Handler

`paddle.train` and `paddle.trainer.XXX.train` take an optional
parameter `event_handler`, which should be either `None` or a function
that handle some events:

1. BeginTraining
1. EndTraining
1. BeginIteration
1. EndIteration
1. BeginPass
1. EndPass

where EndPass is sent if and only if the reader yields
`end_pass=True`.

An example as follows:

```python
def event_handler(event):
    if ininstance(event, paddle.event.EndIteration):
        print paddle.test(...)

paddle.train(topology, parameters, reader, event_handler)
```

If we are writing a PaddlePaddle program in and for iPython/Jypyter,
we can use metaplotlib in the event handler to plot a curve of
cost/error versus iterations, as shown
[here](https://blog.dominodatalab.com/interactive-dashboards-in-jupyter/).

### Distributed Training

If users want to do distributed training on a cluster, s/he should
call `paddle.dist_train` and provides access tokens to the cluster as
a parameter.

For example, if the user has a TLS certificate that allows him to
access a Kubernetes cluster, s/he should be able to call

```python
paddle.dist_train(model,
                  trainer=paddle.trainer.SGD(...,
                                             paddle.updater.Adam(...)),
                  reader=read,
                  k8s_user="yi",
                  k8s_token="kube_cluster_tls.pem",
                  k8s_job="hello",
                  num_parameter_servers=15)
```

The pseudo code of `paddle.dist_train` is as follows:

```python
def dist_train(topology, parameters, trainer, reader, ...):
    if os.getenv("KUBERNETES_SERVICE_HOST") == None:
        image_name = k8s_user + '/' + k8s_job
        docker_build(image_name)
        docker_push()
        kube_ctrl_start_job(image_name, k8s_user, k8s_token)
    else:
        rank = kube_list_containers_in_job_and_return_current_containers_rank()
        if rank == 0:
            master()
        elif rank < 15:
            parameter_server()
        else:
            trainer.train(model, reader=read)
```

Please be aware that if a process is running on the Kubernetes
cluster, it will have some environment variables pre-defined.

If `dist_train` doesn't see these environment variables, it knows
that it's running on users' personal computer, and it should work as a
*launcher*.  Otherwise, it knows that it's running on the cluster and
need to figure out its role as either the master, or a trainer, or a
parameter server.
