# Imperative Programming for Paddlepaddle

## What is Imperative Programming

Take Fluid sudo code as an example:

``` python
x = layer.data("images")
l = layer.data("label")
y = layer.fc(x)
cost = layer.mse(y, l)
optimize(cost)
train(cost, reader=minist.train(), batch_num=1000)
```

As you can see, from line 1 to line 5, we are trying to define a fully connected network with mse as optimizer. please note that variables x, y, l, cost do not point to a real value or tensor, they represent network elements.

if we go imperative, the sudo code would look like this:

```python
for i in xrange(1000):
    x = layer.data("image").next()
    l = layer.data("label").next()
    y = layer.fc(x)
    cost = layer.mse(y, l)
    optimize(cost)
    train(cost)
```

the differences ( fluid v.s. Imperative) are:

1. variables: network elements v.s. runtime variables
1. loops: c++ loop v.s. python loop
1. execution: compile and run v.s. line by line

To summarize it: Compiled execution v.s. Interpreted execution

## Benefits and drawbacks

Imperative programming is getting more and more popular in main stream machine learning frameworks. pyTorch is the one who makes this coding style for machine learning known to all. Tensorflow recently also announced their Eager Execution which is similar.

This is not surprising due to the Debuggability brought by it. Since the major part of the logic is defined and run in the scope of python, you can easily debug your training program in any python IDE. Pause at any step any batch, inspect any variable. This is not possible with current design and implementation of fluid.

The drawback is also obvious when all the heavy liftings are bind with python, Performance.

## How to adopt it

We'd like to adopt this feature while avoiding the performance black hole, so we need to differentiate the imperative mode and our current mode during execution, let's name Fluid's current mode as "conventional mode" for now.

Here are some ultimate targets we'd like to hit when we introduce imperative mode:

1. Conventional mode will still stay the same, "compile and run".
1. Use the same coding style for both modes.
1. Provide a switch to toggle between 2 modes, so that the same piece of code can be executed imperatively (for debug) or conventionally (for performance).
1. No performance compromise in conventional mode.
1. Mode switchability during runtime.

### API updates and challenges

#### The mode switch

We need to provide a mode switch in python API. It would be great  and challenging if the mode can be switched on the fly.

#### Return values

All network node definition (`y = layer.fc(x)`) need to return runtime values in imperative mode.

#### Flow controls

This is a really tricky part, and there is a lot to discuss about. Here are several approaches currently as we discussed:

##### Use native python flow control

Let user use python native `if else`, `while`, etc., so that in imperative mode, there is nothing we need to change, works best with IDEs while debugging.

In conventional mode, when we run the same piece of code, we firstly are going to replace these native controls with Paddle flow control OPs.

the challenging part is not all flow control need to be replaced with Paddle flow control OPs.

##### Use customized python flow control syntax

We are going to provide customized flow control syntaxes like `pd_if` `pd_while`. So that user can still easily debug in IDEs while in imperative mode, and we can also easily target and replace them in conventional mode.

the challenging part is we need to provide a customized python parser.

And there is an issue for both approaches above: python `if` is binary, which means only one branch will be executed in one run, meanwhile, Paddle's `Ifelse_OP` is not.

##### Use Paddle OP

No clue how we can mimic python `if` `else` with the following code, since there is no branching from python point of view:

``` python
ie = pd.ifelse()
with ie.true_block():  // block 1
    d = pd.layer.add_scalar(x, y)
    ie.output(d, pd.layer.softmax(d))
with ie.false_block():  // block 2
    d = pd.layer.fc(z)
    ie.output(d, d+1)
o1, o2 = ie(cond)
```

## Conclusions
As you see, imperative programming does enhances the machine learning coding experience, and it's the feature which made pyTorch so popular among researchers. But this feature does not come cheap, there are a lot of fundamental changes we need to make to our current API. 

## References

1. pyTorch http://pytorch.org/about/
1. Tensorflow Eager Execution: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager 
1. Tensorflow Eager Execution guide: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/g3doc/guide.md
1. Adding new statement to python: https://eli.thegreenplace.net/2010/06/30/python-internals-adding-a-new-statement-to-python/
