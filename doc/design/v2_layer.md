## What are the problems

Paddle V2 API give a flexible way to configure neural network topology. The user can create a neural network topology layer by layer. We use the final layer to represent the neural network topology.

The example code for users is:

```python
img = data_layer(name="img", type=Dense(784))
hidden = fc_layer(input=img, size=200)
prediction = fc_layer(input=hidden, size=10, act=Softmax())
cost = classification_cost(input=prediction, label=data_layer(name="label", type=Integer(10)))

paddle.train(cost, ...)
```

We use `cost` to represent the entire topology of the neural network.  We use the last node to traverse the entire topology by `depth first search` algorithm. The connection between each layer is represented by the `input` parameter.  It is fit for representing a plain neural network topology. However, there are some special layers in Paddle, which are not connected explicitly by `input` parameter. They are:

* Evaluator.  An evaluator is used to compute metrics(such as error rate, f1 score) in Paddle. An evaluator can not be the input of other layers. So we cannot access evaluators by simply traversing back from the final layer.
* Memory. We use memory layers in Recurrent Neural Network; a memory layer represents the output of some layer in the last time step. However, the memory layer connects to its input layer implicitly by sharing the same name. We also cannot traverse to memory layer because maybe there is no layer using this memory layer.
* Recurrent Group.  The recurrent group is a sub-topology config using in Recurrent Neural Network. It represents the layers in each recurrent time-step. We could traverse back to some topology in a recurrent group, but the sub-topology is non-splittable. The recurrent group should be either entirely in the topology or not in the topology. 

## Thinking how to resolve these problems

In our old configuration way, there is a configuration file for each neural network topology. The topology is generated line by line when the configuration function is invoked, no matter whether the output of this function is passed to another method or not.

In another hand, we want a variable hold all topology in paddle.v2. It is impossible to traverse all layers in this topology without any extra information because some layers are not attached to others explicitly.

We also want to use any output variable as a topology. For example, we use `cost` when training and use `prediction` when inference.

In conclusion,

1. We need to add a layer to topology when the layer is created, not by whether the layer is referenced by others.
1. We need to get topology ends with any output variable.
1. We need an extra data structure to hold the information of what layers are created before some variable.

## Implementation

We introduce a concept named `Tape` to hold the order of layer creation. The tape is just an array of layer creation method with arguments. When creating a layer in Paddle, the layer creation method will be stored into the global tape object and the return an index of tape. Each return value of v2 layer method is a simple integer. If we only want to get a part of the topology, we only need to start `playing the tape from the beginning to the desired location`, i.e. just call the stored creation methods one by one until desired end.

The demostration code is here:

```python
class Tape(object):
    def __init__(self):
        self.__creations__ = []

    def append(self, creation):
        self.__creations__.append(creation)
        return len(self.__creations__) - 1

    def call(self, end = -1):
        for i, creation in enumerate(self.__creations__):
            creation()
            if i == end: break

gTape = Tape()

class TapeItem(object):
    # just a helper class, add a to_proto method.
    def __init__(self, idx):
         self.idx = idx
    def to_proto(self):
         gTape.call(self.idx)

def fc_layer_v2(*args, **kwargs):
   return TapeItem(gTape.append(bind(fc_layer_v1, *args, **kwargs)))
```

## advantages and disadvantages of this implementation

The advantages of this implementation are:

1. It make configuration parsing easily to implement and understand. It also fit Paddle's configuration structure and no need to handle some special situation we listed before.
1. It makes error message clear. The call stack of this implementation is not so deep. As a comparison, the depth first search algorithm generates a very deep call stack.

The disadvantages of this implementation are:

### Global Variable

Use a global variable `gTape`. However, the use of this global variable can greatly simplify the implementation and avoid bugs.  In anther hands, in Python, we could uses `with` statement to generate locally `gTape` variable, if the global variable is a big issue.

### Topology with branches

It cannot handle topology with branches. For example

```text
             /---> d --> e
a --> b --> c
             \ ---> f --> g
```

Because the tape is linear, it cannot handle the branch like this. However, there are some solutions for this situation:

1. We could let user decide which node should be skipped when generating topology. It just like playing the tape from begin, but skip some section. The example code shows below:
    ```python
    a = layer(...)
    b = layer(input=a)
    c = layer(input=b)
    d = layer(input=c)
    e = layer(input=d)
    f = layer(input=c)
    g = layer(input=f)
    
    paddle.train(g, skip=[d, e])
    ```

2. We could let user clear the tape. User can define two topology seperately. 
    ```python
    def topology(up_branch=True):
       a = layer(..)
       b = layer(input=a)
       c = layer(input=b)
       if up_branch:
         d = layer(input=c)
         e = layer(input=d)
         return e
       else:
         f = layer(input=c)
         g = layer(input=f)
         return g
  
    with Tape():
        paddle.train(topology(False))
    ```
