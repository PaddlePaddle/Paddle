# Design Doc: PaddlePaddle API

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

We used to think that parameters are part of the toplogy (or layers).
But that is not true, because multiple layers could share the same
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
doens't make sense for each of these threads to have its own copy of
model, because that would duplicate topologies and parameters.
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
multiple simultaneious threads to train a model -- just let each
thread has its own GradientMachine object.
