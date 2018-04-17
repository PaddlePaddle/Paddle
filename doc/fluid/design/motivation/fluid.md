# Design Doc: PaddlePaddle Fluid

## Why Fluid

When Baidu developed PaddlePaddle in 2013, the only well-known open source deep learning system at the time was Caffe.  However, when PaddlePaddle was open-sourced in 2016, many other choices were available. There was a challenge -- what is the need for open sourcing yet another deep learning framework?

Fluid is the answer.  Fluid is similar to PyTorch and TensorFlow Eager Execution, which describes the "process" of training or inference using the concept of a model.  In fact in PyTorch, TensorFlow Eager Execution and Fluid, there is no  concept of a model at all. The details are covered in the sections below. Fluid is currently more extreme in the above mentioned idea than PyTorch and Eager Execution, and we are trying to push Fluid towards the directions of a compiler and a new programming language for deep learning.

## The Evolution of Deep Learning Systems

Deep learning infrastructure is one of the fastest evolving technologies. Within four years, there have already been three generations of technologies invented.

<table>
<thead>
<tr>
<th>Existed since</th>
<th>model as sequence of layers</th>
<th>model as graph of operators</th>
<th>No model</th>
</tr>
</thead>
<tbody>
<tr>
<td>2013 </td>
<td>Caffe, Theano, Torch, PaddlePaddle </td>
<td> </td>
<td> </td>
</tr>
<tr>
<td>2015 </td>
<td> </td>
<td>TensorFlow, MxNet, Caffe2, ONNX, n-graph </td>
<td> </td>
</tr>
<tr>
<td>2016 </td>
<td> </td>
<td>   </td>
<td> PyTorch, TensorFlow Eager Execution, PaddlePaddle Fluid</td>
</tr>
</tbody>
</table>


From the above table, we see that the deep learning technology is evolving towards getting rid of the concept of a model.  To understand the reasons behind this direction, a comparison of the *programming paradigms* or the ways to program deep learning applications using these systems, would be helpful. The following section goes over these.

## Deep Learning Programming Paradigms

With the systems listed as the first or second generation, e.g., Caffe or TensorFlow, an AI application training program looks like the following:

```python
x = layer.data("image")
l = layer.data("label")
f = layer.fc(x, W)
s = layer.softmax(f)
c = layer.mse(l, s)

for i in xrange(1000): # train for 1000 iterations
    m = read_minibatch()
    forward({input=x, data=m}, minimize=c)
    backward(...)

print W # print the trained model parameters.
```

The above program includes two parts:

1. The first part describes the model, and
2. The second part describes the training process (or inference process) for the model.

This paradigm has a well-known problem that limits the productivity of programmers. If the programmer made a mistake in configuring the model, the error messages wouldn't show up until the second part is executed and `forward` and `backward` propagations are performed. This makes it difficult for the programmer to debug and locate a mistake that is located blocks away from the actual error prompt.

This problem of being hard to debug and re-iterate fast on a program is the primary reason that programmers, in general,  prefer PyTorch over the older systems.  Using PyTorch, we would write the above program as following:

```python
W = tensor(...)

for i in xrange(1000): # train for 1000 iterations
    m = read_minibatch()
    x = m["image"]
    l = m["label"]
    f = layer.fc(x, W)
    s = layer.softmax(f)
    c = layer.mse(l, s)
    backward()

print W # print the trained model parameters.
```

We can see that the main difference is the moving the model configuration part (the first step) into the training loop.  This change would allow the mistakes in model configuration to be reported where they actually appear in the programming block.  This change also represents the model better, or its forward pass, by keeping the configuration process in the training loop.

## Describe Arbitrary Models for the Future

Describing the process instead of the model also brings Fluid, the flexibility to define different non-standard models that haven't been invented yet.

As we write out the program for the process, we can write an RNN as a loop, instead of an RNN as a layer or as an operator.  A PyTorch example would look like the following:

```python
for i in xrange(1000):
    m = read_minibatch()
    x = m["sentence"]
    for t in xrange x.len():
        h[t] = the_step(x[t])
```        

With Fluid, the training loop and the RNN in the above program are not really Python loops, but just a "loop structure" provided by Fluid and implemented in C++ as the following:

```python
train_loop = layers.While(cond)
with train_loop.block():
  m = read_minibatch()
  x = m["sentence"]
  rnn = layers.While(...)
  with rnn.block():
    h[t] = the_step(input[t])
```    

An actual Fluid example is described  [here](https://github.com/PaddlePaddle/Paddle/blob/bde090a97564b9c61a6aaa38b72ccc4889d102d9/python/paddle/fluid/tests/unittests/test_while_op.py#L50-L58).

From the example, the Fluid programs look very similar to their PyTorch equivalent programs, except that Fluid's loop structure, wrapped with Python's `with` statement, could run much faster than just a Python loop.

We have more examples of the [`if-then-else`](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/execution/if_else_op.md) structure of Fluid.

## Turing Completeness

In computability theory, a system of data-manipulation rules, such as a programming language, is said to be Turing complete if it can be used to simulate any Turing machine.  For a programming language, if it provides if-then-else and loop, it is Turing complete.  From the above examples, Fluid seems to be Turing complete; however, it is noteworthy to notice that there  is a slight difference between the `if-then-else` of Fluid and that of a programming language. The difference being that the former runs both of its branches and splits the input mini-batch into two -- one for the True condition and another for the False condition. This hasn't been researched in depth if this is equivalent to the `if-then-else` in programming languages that makes them Turing-complete.  Based on a conversation with [Yuang Yu](https://research.google.com/pubs/104812.html), it seems to be the case but this needs to be looked into in-depth.

## The Execution of a Fluid Program

There are two ways to execute a Fluid program.  When a program is executed, it creates a protobuf message [`ProgramDesc`](https://github.com/PaddlePaddle/Paddle/blob/a91efdde6910ce92a78e3aa7157412c4c88d9ee8/paddle/framework/framework.proto#L145) that describes the process and is conceptually like an [abstract syntax tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree).

There is a C++ class [`Executor`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/executor.h), which runs a `ProgramDesc`, similar to how an interpreter runs a Python program.

Fluid is moving towards the direction of a compiler, which is explain in [fluid_compiler.md](fluid_compiler.md).

## Backward Compatibility of Fluid

Given all the advantages from the removal of the concept of a *model*, hardware manufacturers might still prefer the existence of the concept of a model, so it would be easier for them to support multiple frameworks all at once and could run a trained model during inference.  For example, Nervana, a startup company acquired by Intel, has been working on an XPU that reads the models in the format known as [n-graph](https://github.com/NervanaSystems/ngraph).  Similarly, [Movidius](https://www.movidius.com/) is producing a mobile deep learning chip that reads and runs graphs of operators.  The well-known [ONNX](https://github.com/onnx/onnx) is also a file format of graphs of operators.

For Fluid, we can write a converter that extracts the parts in the `ProgramDesc` protobuf message, converts them into a graph of operators, and exports the graph into the ONNX or n-graph format.
