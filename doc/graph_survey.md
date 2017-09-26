## Survey on Graph

神经网络框架通常提供Symbolic的接口给用户，来方便的书写网络配置。这里主要调研一下不同神经网络中框架中，用户书写的配置（等号左边）与最终解析得到的Graph之间的关系。

### Mxnet

用户配置网络的核心概念是`Symbol`，Mxnet在C++端实现了`Symbol`，并通过CAPI暴露到Python端。在这里可以参考Mxnet中对`Symbol`的注释：

`Symbol` is help class used to represent the operator node in Graph.
`Symbol` acts as an interface for building graphs from different components like Variable, Functor and Group. `Symbol` is also exported to python front-end (while Graph is not) to enable quick test and deployment. Conceptually, symbol is the final operation of a graph and thus including all the information required (the graph) to evaluate its output value.


一个简单的网络定义如下：

```python
def get_symbol(num_classes=10, **kwargs):
    data = mx.symbol.Variable('data')
    data = mx.sym.Flatten(data=data)
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=num_classes)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp
```


需要注意的是，这里的Variable实际上也是一个Symbol。每个基本Symbol最终会对应到一个Node，每个Node都有对应的属性attr，attr中有一个字段为op。当这个Symbol表示Varaible时（通常是输入数据），attr中的op字段为空。

Symbol包含的成员变量为std::vector<NodeEntry> outputs，NodeEntry中包含一个指向Node的指针。


Mxnet的Symbol可以绑定到一个Executor上，在解析为Graph之后，得以执行。



### TensorFlow

用户配置网络的核心概念是`Tensor`，在Python端定义了`Tensor`，在这里可以直接参考TensorFlow对Tensor的注释：


A `Tensor` is a symbolic handle to one of the outputs of an `Operation`. It does not hold the values of that operation's output, but instead provides a means of computing those values in a TensorFlow @{tf.Session}.

一个简单的使用样例如下：

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

  
Tensor的一些主要成员变量和接口可以参考如下：

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

TensorFlow的Tensor可以作为target被session来run，实际上是Tensor已经包含了所有的Graph信息，可以track data dependency。


### Dynet

用户配置网络的核心概念是`Expression`，在C++端定义了`Expression`。用户通过书写Expression来完成Graph的构建。

一个简单的使用样例如下：

```cpp
ComputationGraph cg;
Expression W = parameter(cg, pW);

Expression in = input(cg, xs[i]);
Expression label = input(cg, ys[i]);
Expression pred = W * in;
Expression loss = square(pred - label);
```

需要注意的是，输入数据以及参数也同样使用Expression来书写。每个Expression对应一个Node，输入数据也对应一个Node。

Expression的主要成员为ComputationGraph，可以在用户配置网络的过程中修改Graph。Expression同样可以被作为目标来执行，因为Expression中已经包含了所有的依赖关系。


### 总结

实际上Mxnet/TensorFlow/Dynet中的Symbol/Tensor/Expression是同一个层级的概念，我们暂时统一这个概念的名称为Expression，这层概念有如下几个特点：

- 在用户配置网络时，所有的返回值都是Expression，包括最初的输入数据，及参数等
- Expression已经包含了所有的依赖关系，可以被当做执行的target
