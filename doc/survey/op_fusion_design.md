# Operator fusion  
Fusing multiple operators together is an important method to optimize the program execution, particularly for GPU or other specialized accelerators. An obvious benefit is to avoid the overhead of saving the intermediate result back into global memory.   

There are generally two ways to fuse operators, fusing directly connected operators and fusing non directly connected operators. The first method is mainly used by [NNVM Compiler](https://github.com/dmlc/tvm/) and [XLA](https://www.tensorflow.org/performance/xla/). The second method is mainly used by Dynet and TensorFlow Fold to do auto-batching. The principle of fusing operator is according to some rules to combine multiple operations into one, for example, `Y = X * W` and `Z = Y + B` can be fused to `Z = X * W + B`, and `Y1 = X1 * W` and `Y2 = X2 * W` can be fused to `[Y1;Y2] = [X1;X2] * W`. In order to get a short-term profit, we decided to try to manually specify these rules.   

## Challenge
The challenge of fusing operators is:
  - how to make the rules.
  - how to implement these rules efficiently.

### How to make the rules?

The problem of determining the best single location for a fusion operator is an NP-hard combinatorial problem. After analysis the operators of the DL model, we found there are two group of operators can be fused explicitly, one is the simple and adjacent operations, for example, `tmp = x + y` and `z = Relu(tmp)`, and the other is the operators that have the same function, for example, a serials of `SGD` or `Momentum`. They usually appear in the model in a large number. So we should think about how to fuse them separately first.

### How to implement these rules efficiently?
#### How to fuse the adjacent operations efficiently?
Here we use a template function to represent the fused operations. The pros of using a template function are that it is simple and efficient, and the cons are that it is not easy to expand, and it can only be used to express some simple operations. So taking into account our current needs, the template function is more appropriate.

#### How to fuse the operators that have the same function efficiently?
We take SGD operator as an example, the training model may have hundreds of parameters and correspondingly have the same number of SGD operators. The expression(`w = w - lr*w_g`) of those operators is the same, so during of training, the executor will execute this expression hundreds time in CPU or other specialized accelerators. If we can fuse them and make the address of all `w` and all `w_g` continuous respectively, we only need execute one time. For some accelerators, the time of launching kernel is not neglected, so the time of hundreds of times of launching and executing kernel may be larger than launching and executing only once. There usually are many operators that similar to `SGD` in the DL model, such as `AllReduce` and `FC`.
