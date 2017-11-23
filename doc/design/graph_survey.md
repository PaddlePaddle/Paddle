## Survey on Graph

Neural network framework often provides symbolic API for users to write network topology conveniently. This doc manily focus on symbolic API in most popular neural network frameworks, and try to find out how to parse symbolic configuration to a portable file, such as protobuf or json.

### Mxnet

The core concept of symbolic API is `Symbol`. Mxnet implements `Symbol` class in C++, and export to Python using C-API. Please refer to the comments in Mxnet:


`Symbol` is help class used to represent the operator node in Graph.
`Symbol` acts as an interface for building graphs from different components like Variable, Functor and Group. `Symbol` is also exported to python front-end (while Graph is not) to enable quick test and deployment. Conceptually, symbol is the final operation of a graph and thus including all the information required (the graph) to evaluate its output value.


A simple network topology wrote by Symbol is as follows:

```python
def get_symbol(num_classes=10, **kwargs):
    data = mx.symbol.Variable('data')
    data = mx.symbol.Flatten(data=data)
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=num_classes)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp
```



Varible here is actually a Symbol. Every basic Symbol will correspond to one Node, and every Node has its own NodeAttr. There is a op field in NodeAttr class, when a Symbol represents Variable(often input data), the op field is null.

Symbol contains a data member, std::vector<NodeEntry> outputs, and NodeEntry cantains a poniter to Node. We can follow the Node pointer to get all the Graph.

And Symbol can be saved to a Json file.

Here is a detailed example:

```
>>> import mxnet as mx
>>> data = mx.symbol.Variable('data')
>>> print data.debug_str()
Variable:data

>>> data = mx.symbol.Flatten(data=data)
>>> print data.debug_str()
Symbol Outputs:
	output[0]=flatten0(0)
Variable:data
--------------------
Op:Flatten, Name=flatten0
Inputs:
	arg[0]=data(0) version=0

>>> fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
>>> print fc1.debug_str()
Symbol Outputs:
	output[0]=fc1(0)
Variable:data
--------------------
Op:Flatten, Name=flatten0
Inputs:
	arg[0]=data(0) version=0
Variable:fc1_weight
Variable:fc1_bias
--------------------
Op:FullyConnected, Name=fc1
Inputs:
	arg[0]=flatten0(0)
	arg[1]=fc1_weight(0) version=0
	arg[2]=fc1_bias(0) version=0
Attrs:
	num_hidden=128

```


### TensorFlow


The core concept of symbolic API is `Tensor`. Tensorflow defines `Tensor` in Python. Please refer to the comments in TensorFlow:

A `Tensor` is a symbolic handle to one of the outputs of an `Operation`. It does not hold the values of that operation's output, but instead provides a means of computing those values in a TensorFlow [Session](https://www.tensorflow.org/api_docs/python/tf/Session).

A simple example is as follows:

```python
  # Build a dataflow graph.
  c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
  d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
  e = tf.matmul(c, d)

  # Construct a `Session` to execute the graph.
  sess = tf.Session()

  # Execute the graph and store the value that `e` represents in `result`.
  result = sess.run(e)
```

  
The main method of `Tensor` is as follows: 
 
 
```python
@property
def op(self):
  """The `Operation` that produces this tensor as an output."""
  return self._op

@property
def dtype(self):
   """The `DType` of elements in this tensor."""
  return self._dtype

@property
def graph(self):
  """The `Graph` that contains this tensor."""
  return self._op.graph

@property
def name(self):
  """The string name of this tensor."""
  if not self._op.name:
    raise ValueError("Operation was not named: %s" % self._op)
  return "%s:%d" % (self._op.name, self._value_index)

@property
def device(self):
  """The name of the device on which this tensor will be produced, or None."""
  return self._op.device
```


Tensor can be taken as target to run by session. Tensor contains all the information of Graph, and tracks data dependency.


Here is a detailed example:


```
>>> import tensorflow as tf
>>> c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
>>> print c.graph
<tensorflow.python.framework.ops.Graph object at 0x10f256d50>
>>> d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
>>> print d.graph
<tensorflow.python.framework.ops.Graph object at 0x10f256d50>
>>> e = tf.matmul(c, d)
>>> print e.graph
<tensorflow.python.framework.ops.Graph object at 0x10f256d50>
```

### Dynet


The core concept of symbolic API is `Expression`, and Dynet defines `Expression` class in C++.


A simple example is as follows:

```cpp
ComputationGraph cg;
Expression W = parameter(cg, pW);

Expression in = input(cg, xs[i]);
Expression label = input(cg, ys[i]);
Expression pred = W * in;
Expression loss = square(pred - label);
```

The input data and parameter are also represented by Expression. Every basci Expression corresponds to a Node. And input data is also a Node. 

Expression has a data member ComputationGraph, and ComputationGraph will be modified in users' configuring process. Expression can be a running target, beacuse Expression contains all dependency.


Here is a detailed example:

write topology in C++

```
ComputationGraph cg;
Expression W = parameter(cg, pW);
cg.print_graphviz();

Expression pred = W * xs[i];
cg.print_graphviz();

Expression loss = square(pred - ys[i]);
cg.print_graphviz();
```

compile and print

```
# first print
digraph G {
  rankdir=LR;
  nodesep=.05;
  N0 [label="v0 = parameters({1}) @ 0x7ffe4de00110"];
}
# second print
digraph G {
  rankdir=LR;
  nodesep=.05;
  N0 [label="v0 = parameters({1}) @ 0x7ffe4de00110"];
  N1 [label="v1 = v0 * -0.98"];
  N0 -> N1;
}
# third print
digraph G {
  rankdir=LR;
  nodesep=.05;
  N0 [label="v0 = parameters({1}) @ 0x7ffe4de00110"];
  N1 [label="v1 = v0 * -0.98"];
  N0 -> N1;
  N2 [label="v2 = -1.88387 - v1"];
  N1 -> N2;
  N3 [label="v3 = -v2"];
  N2 -> N3;
  N4 [label="v4 = square(v3)"];
  N3 -> N4;
}
```

### Conclusion


Actually, Symbol/Tensor/Expression in Mxnet/TensorFlow/Dynet are the same level concepts. We use a unified name Expression here, this level concept has following features:

- Users wirte topoloy with symbolic API, and all return value is Expression, including input data and parameter.
- Expression corresponds with a global Graph, and Expression can also be composed.
- Expression tracks all dependency and can be taken as a run target
