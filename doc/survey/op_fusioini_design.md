# Operator fusion  
Fusing multiple operators together is an important method to optimize the program execution, particularly for GPU or other specialized accelerators. An obvious benefit is avoiding the overhead of saving the intermediate result back into global memory.   

There are generally two ways to fuse operators, fusing directly connected operators and fusing non directly connected operators. The first method is mainly used by [NNVM Compiler](https://github.com/dmlc/tvm/) and [XLA](https://www.tensorflow.org/performance/xla/). The second method is mainly used by Dynet and TensorFlow Fold to do auto-batching. The principle is to fuse several Op in the graph according to the certain rules, for example, `Y = X * W` and `Z = Y + B` can be fused to `Z = X * W + B`, and `Y1 = X1 * W` and `Y2 = X2 * W` can be fused to `[Y1;Y2] = [X1;X2] * W`. In order to get a short-term profit, we decided to try to manually specify these rules.   

The challenge of fusing operators is:
  - how to make the rules.
  - how to implement these rules efficiently.

The problem of determining the best single location for a fusion operator is an NP-hard combinatorial problem. After analysis the operators of DL model, we found there are two group of operators can be fused explicitly, one is the operators that have the same function, for example, a serials of `SGD` or `Momentum`, the other is the simple and adjacent operations, for example, `tmp = x + y` and `z = Relu(tmp)`. And they usually appear in the model in a large number. So we should think about how to fuse them separately first.
