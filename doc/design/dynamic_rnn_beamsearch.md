# Design of Dynamic RNN with Beamsearch
With a `while-loop` and an array of states, one can implement a RNN model in Python easily,
while how to build a RNN using Paddle operators in the similar intuitive way is a question.

One of the mature DL platforms, TensorFlow, offers a mass of operators for mathematical and conditional operations,
in that way, TF users can simulate the Python-way and uses some dynamic RNN, encoder or decoder to build a RNN model represented as a graph and the model supports large-scale computation.

That is a good way to follow, by turning a complex algorithm into the combination of several highly-reusable operators, make our framework more flexible.

In this documentation, we will first give a glimpse of the dynamic encoder and dynamic decoder in TensorFlow, then propose a general scheme of our dynamic RNN's design.

## Dynamic RNN in TensorFlow
Reference to [dynamic_rnn](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/rnn.py#L443)
