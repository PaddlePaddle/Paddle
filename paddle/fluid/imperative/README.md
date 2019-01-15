# Overview

Imperative Programming is easier to learn, debug and try new ideas.

# Related Works

## Pytorch
https://pytorch.org/

## TensorFlow Eager
https://www.tensorflow.org/guide/eager

# Design

## API
```python
class Layer(object):

  def __call__(inputs):
    # build some parameter once.
    # ...
    return self.apply(inputs):

  def forward(inputs):
    # forward logic with paddle operators. backward auto-generated.


class PyLayer(core.PyLayer):

  def __call__(cls, inputs):
    # trace the logic.

  @staticmethod
  def forward(inputs):
    # any forward logic implemented with numpy io.

  @staticmethod
  def backward(inputs):
    # any backward logic implemented with numpy io.



```


## Tracer

Current: Python Variable -> C++ VarBase -> C++ Variable -> C++ Tensor

Longer term.
```python

# Parent class.
class PyVarBase(object):
  pass

# Current python variable.
class Variable(PyVarBase):
  pass

class IVariable(PyVarBase):
  def __init__(self):
    self._ivar = core.VarBase()

  def to(device): pass
  def value(): pass
  def backward(): pass
  def gradient_value(): pass
  # operators to override.
```



```cpp
class Tracer {
 public:
  explicit Tracer(framework::BlockDesc* root_block) : root_block_(root_block) {}

  virtual ~Tracer() {}

  void Trace(OpBase* op,
             const std::map<std::string, std::vector<VarBase*>>& inputs,
             const std::map<std::string, std::vector<VarBase*>>& outputs,
             framework::BlockDesc* block, const bool stop_gradient = false);

  std::vector<VarBase*> PyTrace(OpBase* op, const std::vector<VarBase*>& inputs,
                                bool stop_gradient = false);
};
```

* Trace forward operations
* Perform simple python level infer and return to user.
* Perform autograd to generate gradients.
* Clear trace.
* Apply gradients with optimizers

## Autodiff

Lots of research already.
https://autodiff-workshop.github.io/
https://en.wikipedia.org/wiki/Automatic_differentiation

## Execution Engine

Lazy execution of pushed C++ operations.

## Tests

* All op tests run once in static graph, once in imperative mode.

## Refactor

* All function layers with parameters converted to class Layers.
* Models converted to imperative mode.

# Examples

```python
class MyLayer(fluid.imperative.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()

    def forward(self, inputs):
        x = fluid.layers.relu(inputs)
        x = fluid.layers.elementwise_mul(x, x)
        x = fluid.layers.reduce_sum(x)
        return [x]


class MyPyLayer(fluid.imperative.PyLayer):
    def __init__(self):
        super(MyPyLayer, self).__init__()

    @staticmethod
    def forward(inputs):
        return np.tanh(inputs[0])

    @staticmethod
    def backward(inputs):
        return np.array(dout) * (1 - np.square(np.array(out)))


class MLP(fluid.imperative.Layer):
    def __init__(self):
        super(MLP, self).__init__()
        self._fc1 = FC(3,
                       fluid.ParamAttr(
                           initializer=fluid.initializer.Constant(value=0.1)))
        self._fc2 = FC(4,
                       fluid.ParamAttr(
                           initializer=fluid.initializer.Constant(value=0.1)))

    def forward(self, inputs):
        x = self._fc1(inputs)
        x = self._fc2(x)
        x = fluid.layers.reduce_sum(x)
        return x


 np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
 with fluid.imperative.guard():
     var_inp = fluid.imperative.base.to_variable(np_inp)
     mlp = MLP()
     out = mlp(var_inp)
     dy_out = out._numpy()
     out._backward()
```


## Save/Load Models

TODO

# Plan

2.1，3 fulltime, Can run a few simple models. (Currently, 2 20% engs)

4.1, 4 fulltime, Can run 6 models, Performance 70% Pytorch. Release alpha.

6.1, 5 fulltime, Performance close to Pytorch, can run multi-devices. Release Beta.

8.1, 5 fulltime, Works in general. Update existing models. Can compile to static graph, support more optimizations.

12.1 Done.

# Discussion

TODO.
