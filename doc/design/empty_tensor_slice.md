### Empty Tensor Design

I think we may need a special tensor "Empty Tensor" in our tensor design.
In our current codes, it will throw error when we have dim = {0,3,4}.

However, in case of switch_op, if_else_op and so forth, it is very possible that one branch has nothing 
and everything goes to only one branch.

In Caffe2, they have .

### Slice Tensor Design
if a tensor is of shape {1,3,4}, it does not support slice.
https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h
