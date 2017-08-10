### Empty Tensor Design

I think we may need a special tensor "Empty Tensor" in our tensor design.
In our current codes, it will throw error when we have dim = {0,3,4}.

However, in case of switch_op, if_else_op and so forth, it is very possible that one branch has nothing 
and everything goes to only one branch.

In Caffe2, they have:
  /**
   Returns a const raw void* pointer of the underlying storage. mutable_data()
   or raw_mutable_data() must have been called prior to this function call.
   */
  inline const void* raw_data() const {
  
    
    CAFFE_ENFORCE_WITH_CALLER(data_.get() || size_ == 0);
    
    return data_.get();
  }

### Slice Tensor Design
if a tensor is of shape {1,3,4}, it does not support slice.
https://github.com/caffe2/caffe2/blob/master/caffe2/operators/utility_ops.h


PADDLE_ENFORCE_NE(dims_[0], 1, "Can not slice a tensor with dims_[0] = 1.");
I think we may not really need to actually enforce this. right? :)
if it is dimension 1, we need to slice the whole thing.
Otherwise, in if/else/switch, will be a lot of boundary case checks....

