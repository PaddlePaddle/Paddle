# Design Doc: New Paddle API

To write a Paddle program using the current API, we would have to write two Python source files -- one defines the data provider and the other defines the network or run the for loop.  This doesn't work well with Notebooks.  So we decide to redesign the API.  This document describes the basic design concerns.

## Basic Concepts

The API design depends on basic concepts about deep learning.

### Model

For deep learning, a model includes two parts: the topology and parameters.  Currently, the concept *model* in Paddle contains only the topology, and parameters are in another concept *gradient machine*.  This differs from the intuition and makes it difficult to save/load models.  In this design, we should keep both topology and parameters in a *model*.
   
Algorithms like GAN requires the flexibility to temporarily compose multiple models into one, while keeping each of them workable alone.  We will show later that we don't need model composite API; instead, we can use composite gradient machines.

### Gradient Machine and Updater

In order to learn the model, we need to run the error backpropagation algorithm iteratively.  In each iteration, we run the forward algorithm with a minibatch of data.  This updates the outputs of layers.  Then we run a backward algorithm which computes the gradients of every parameter.  The outputs and gradients are not part of the model; instead, they are side effects of the training process and should be maintained in the trainer.
   
After the invocation of the backward algorithm, we should update model parameters using the gradients and parameters like learning rate.  This step might involve communications with the parameter server in the case of distributed training.  This complexity motivates us to separate the trainer into two concepts:
   
1. *gradient machine*, which computes and maintains layer outputs and gradients, and
      
1. *updater*, which encapsulates the updating algorithm.
   
It seems that *cost function* is a property of *gradient machine*?
   
### Data

Models are trained using data sets.  We hope to provide a set of utility data sets encapsulated in Python packages like `paddle.data.amazon.product_review` and `paddle.data.wikipedia.articles`.  A reasonable idea might be that in each of these packages, we provide a `new` function that returns a reader object or a Python iterator.  And the *reader* has a read method `read`, which, once called, returns a minibatch and a flag indicating if it reaches the end of a data set.  For online learning, the flag would always be False.  In this way, we don't have to have two concepts: *pass* and *iteration*; instead, we need only the latter.

## Examples

### A Simple Network

```python
gm = paddle.gradient_machine.new(model)  # gm uses default cost function.
ud = paddle.updater.new_simple()  # A simple updater doesn't work with parameter servers.
rd = paddle.data.amazon.product_review.new()
mt = paddle.metric.new_acc()

for mb, pass_end in rd.read():
    gm.feed(mb)
    ud.update(gm)
    mt.append(gm, mb) # compute and record the accuracy of this minibatch.
    if pass_end:
        log(mt.flash()) # output aggregated accuracy records of this pass and reset mt.
```

In this example, `GradientMachine.feed` is a convenience that calls `GradientMachine.forward` and `GradientMachine.backward`.

### A GAN Example

```python
input_gen = paddle.layer.input(...)
hidden_gen = paddle.layer.fc(input_gen, ...)
output_gen = paddle.layer.fc(hidden_gen, ...)

# gm_gen maintains layer outputs and gradients of model gen.
gm_gen = paddle.gradient_machine.new(output_gen)

input_dis = paddle.layer.input(...)
hidden_dis = paddle.layer.fc(intput_dis, ...)
output_dis = paddle.layer.softmax(hidden_dis, ...)

# gm_dis maintains layer outputs and gradients of model dis.
gm_dis = paddle.gradient_machine.new(output_dis)

# gm_comp maintains layer outputs and gradients of both gen and dis.
gm_comp = paddle.gradient_machine.compose(output_gen, output_dis)

ud = paddle.updater.new_simple()

rd = paddle.data.mnist.new()

for mb, pass_end in rd.read():
    fake = gm_gen.forward(mb)
    fake_label = paddle.input.new(False, ...)
    real_label = paddle.input.new(True, ...)
    gm_dis.feed([fake, fake_label])
    gm_dis.feed([mb, real_label])
    ud.update(gm_dis)
    
    gm_comp.feed([mb, real_label])
    ud.update(gm_comp, output_gen) # updates only the model whose output layer is output_gen.
```

A key point here is that we use the output layer to indicate a model.  I think that we can achieve this as long as each layer knows about its predecessors so that we can trace from the output layer upward till the input layers.  Please be aware that we didn't compose two models in above example code; instead, we only created a gradient machine which covers both `model_gen` and `model_dis`.
