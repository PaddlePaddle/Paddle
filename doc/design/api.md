# PaddlePaddle API

## Ingredients

As the first step of our design, we list important concepts in deep
learning and try to figure their relationship, as shown below:

```
Model = {topology, parameters}

Evaluator = {Model*, activations}
- forward
- test

GradientMachine = {Model*, gradients}
- backward

Optimizer = {Model*, Evaluator*, GradientMachine*}
- train
- update
- checkpoint
```

where the pair of curly braces `{` and `}` indicate *composition*, `*`
indicates a *reference*, and `-` marks a "class method".


### Model

We used to think that parameters are part of the topology (or layers).
But that is not true because multiple layers could share the same
parameter matrix.  An example is a network that compares two text
segments in a semantic space:

```
          semantic
text A -> projection ---\
          layer A        \
                          cosine
                          similarity -> output
                          layer
          semantic       /
text B -> projection ---/
          layer B
```

In this network, the two semantic projection layers (A and B) share
the same parameter matrix.

For more information about our API that specifies topology and
parameter sharing, please refer to [TODO: API].


### Evaluator

Supposed that we have a trained ranking model, we should be able to
use it in our search engine.  The search engine's Web server is a
concurrent program so to serve many HTTP requests simultaneously.  It
doesn't make sense for each of these threads to have its own copy of the model because that would duplicate topologies and parameters.
However, each thread should be able to record layer outputs, i.e.,
activations, computed from an input, derived from the request.  With
*Evaluator* that saves activations, we can write the over-simplified
server program as:

```python
m = paddle.model.load("trained.model")

http.handle("/",
            lambda req:
                e = paddle.evaluator.create(m)
                e.forward(req)
                e.activation(layer="output")) # returns activations of layer "output"
```

### GradientMachine

Similar to the evaluation, the training needs to compute gradients so
to update model parameters.  Because an [optimizer](#optimizer) might
run multiple simultaneous threads to update the same model, gradients
should be separated from the model.  Because gradients are only used
in training, but not serving, they should be separate from Evaluator.
Hence the `GradientMachine`.

### Optimizer

None of Model, Evaluator, nor GradientMachine implements the training
loop, hence Optimizer.  We can define a concurrent optimizer that runs
multiple simultaneous threads to train a model -- just let each
thread has its own GradientMachine object.

Most models should be able to be trained using the
`paddle.optimizer.SGD` by calling its `train` method.  Many
customizations to the SGD algorithm happens with the update equation,
e.g., momentum and the Adam SGD algorithm.  We make `train` calls
`update` to do an update, so that we can derive a `paddle.optimizer.Adam`
from `paddle.optimizer.SGD` by overrides only the `update` method.


## Programming

A fictive example of PaddlePaddle program looks like the following:

```python
import paddle

def read(args):
    f = open_file(args["filename"])
    mb = read_a_minibatch(f)
    end_pass = eof(f)
    if end_pass:
       f = open_file(args["filename"]) # rewind for reading again
    yield mb, end_pass

input = paddle.layer.data(...)
intermediate = paddle.layers.fc(input)
output = paddle.layer.softmax(intermediate)

model = paddle.model.create(output)

paddle.train(model, data_provider=read)
```

This shows some important part of a program:

1. Define how to read (and augment) data by defining a function, in
   this example, `read`, that `yields` a minibatch and a boolean flag
   `eof_of_pass`.

1. Define the topology, `input`, `intermediate`, and `output` in this
   example.

1. Create parameters from the topology thus forms the model by calling
   `paddel.model.create`.

1. Train the model by calling `paddle.train`.


### Reader

Not all programming frameworks allow users to define I/O functions.
An example is Google MapReduce, which can only read from text,
SSTable, and RecordIO files.  Hadoop MapReduce allows users to define
readers and writers by deriving from base classes `Reader` and
`Writer`.  The former is less flexible but also less error-prone.  We
decide to provide the flexibility to users to define their readers.


#### A Synthetic Data Reader

Sometimes we want to test a topology and/or a training algorithm using
synthetic data.  We can do this by defining the reader a synthesizer:

```python
def read(args):
    x = sample_from_uniform(0.0, 1.0)
    y = sample_from_gauss(2 * x, sigma)
    yield {x, y}, False # no end-of-file so no end-of-pass
```

#### A Reader for Online Learning

Readers can also read an infinite data stream, e.g., a log stream from
a search engine and collected by Kafka:

```python
def read(args):
    log_stream = kafka.open_channel(args["kafka channel name"])
    yeild log_stream.read(), False # no end-of-pass in online learning
```

### Topology

By default, layers don't have names.  But if we want to refer to a
layer later some time, for example, when we do serving using the model
and wants activations/outputs of a layer, we should give it a name.

```python
input = paddle.layer.data(...)
intermediate = paddle.layer.fc(input, name="inter", ...)
output = paddle.layer.softmax(intermediate, name="output", ...)

m = paddle.model.create(output)
e = paddle.evaluator.create(model)
e.forward(read_an_input()) # compute activations of all layers.
print e.activations(layer="inter")  # retrieve the activations of layer "inter"
print e.activations(layer="output") # retrieve the activations of layer "output"
```

#### Sharing Parameters

In [above section](#model) we shows a network whose two layers share
the same parameter matrix.  To specify such cases, we give "parameter
names" to layers.  If some layers have the same paraemter names,
`paddle.model.create` creates a single parameter matrix for these
layers:

```python
text1 = paddle.layer.data(...)
sematic1 = paddle.layer.fc(text1, ..., parameter_name="sematic_projection")
text2 = paddle.layer.data(...)
sematic2 = paddle.layer.fc(text2, ..., parameter_name="sematic_projection")
out = paddle.layer.cosine(semantic1, semantic2)
```

We can also share parameter matrices between layers in different
models.  To do this, we need an additional parameter that refers to a
model:

```python
model1_input = paddle.layer.data(...)
model1_output = paddle.layer.softmax(model1_input, ...,
                                     parameter_name="a_parameter_matrix")
model1 = paddle.model.create(model1_output)

# Another model
model2_semantic = paddle.layer.fc(text2, ...,
                                  parameter_name="a_parameter_matrix",
                                  parameter_model=model1)
```

### Training

The recommended way to training a model is to call `paddle.train`,
which simply calls `paddle.optimizer.Default`, a global variable of
type `paddle.optimizer.SGD`.  Equivalently, we can do

```python
opt = paddle.optimizer.SGD(...)
opt.train(model, reader=read, ...)
```

#### Distributed Training

If users want to do distributed training on a cluster, s/he should
call `paddle.dist_train` and provides access tokens to the cluster as
a parameter.

For example, if the user has a TLS certificate that allows him to
access a Kubernetes cluster, s/he should be able to call

```python
paddle.dist_train(model,
                  reader=read,
                  optimizer=paddle.optimizer.SGDOptimizer(...),
                  k8s_user="yi",
                  k8s_token="kube_cluster_tls.pem",
                  k8s_job="hello",
                  num_parameter_servers=15)
```

The pseudo code if `paddle.dist_train` is as follows:

```python
def dist_train():
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
            optimizer.train(model, reader=read)
```

Please be aware that if a process is running on the Kubernetes
cluster, it will have some environment variables pre-defined.

If `dist_train` doesn't see these environment variables, it knowns
that it's running on users' personal computer, and it should work as a
*launcher*.  Otherwise, it knows that it's running on the cluster and
need to figure out its role as either the master, or a trainer, or a
parameter server.
