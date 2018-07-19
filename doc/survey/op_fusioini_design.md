# Operator fusion  
Fusing multiple operators together is an important method to optimize the program execution, particularly for GPU or other specialized accelerators. An obvious benefit is fusing multiple operators can avoid the overhead that saving the intermediate result back into global memory.   

There are generally two ways to fuse operators, fusing directly connected operators and fusing non directly connected operators. The first method is mainly used by NNVM Compiler and XLA. The second method is mainly used by Dynet and TensorFlow Fold to do auto-batching. The principle is to fuse several Op in the graph according to certain rules, for example: `Y = X * W` and `Z = Y + B` can be fused to `Z = X * W + B`, and `Y1 = X1 * W` and `Y2 = X2 * W` can be fused to `[Y1;Y2] = [X1;X2] * W`. In order to get a short-term profit, we decided to try to manually specify these rules.   

The challenge of fusing operators is:
  - how to make the rules.
  - how to implement these rules efficiently.

## An Overview
