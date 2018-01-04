# Error Clip

## Overview

Error clip is widely used in model training to prevent gradient exploding. It takes a value as clip threshold. With error clip, all gradient values will be checked before they are taken by the next `grad_op`, and values greater than the threshold will be clipped.

## Usage

Users can enable clip and set related attributes via invoking `Optimizer`'s `minimize` API:

```python
def minimize(self,
             loss,
             startup_program=None,
             parameter_list=None,
             no_grad_set=None,
             error_clip=None):
    # ...
```

The default value of `error_clip` is `None`, which means no error clip is employed. When it's not `None`, it should take an object of `BaseErrorClipAttr`'s derived class. So far, `BaseErrorClipAttr` has only one derived class: `ErrorClipByValue`, whose constructor is:

```python
ErrorClipByValue(max, min=None)
```

`max` and `min` represent the maximal and minimal clip threshold respectively. When the `min` is None, the minimal threshold will be assigned with `-max`.

So we can enable the error clip with threshold `[-5.0, 5.0]` by:

```python
opt = fluid.optimizer.SGD(learning_rate=0.001)
opt.minimize(loss=avg_cost, error_clip=ErrorClipByValue(max=5.0))
```

## Implementation

The `BaseErrorClipAttr` and its derived class `ErrorClipByValue` are defined in *clip.py*.

```python
class BaseErrorClipAttr(object):
    def create_clip_op_desc(self, grad_name):
        raise NotImplementedError()

    def prepend_clip_op_desc(self, op_descs):
        grad_names = set()
        for op_desc in op_descs:
            grad_names.update(filter(lambda n: n.find(
                core.grad_var_suffix()) != -1, op_desc.output_arg_names()))
        for n in grad_names:
            op_descs.append(self.create_clip_op_desc(grad_name=n))


class ErrorClipByValue(BaseErrorClipAttr):
    def __init__(self, max, min=None):
        max = float(max)
        if min is None:
            min = -max
        else:
            min = float(min)
        self.max = max
        self.min = min

    def create_clip_op_desc(self, grad_name):
        desc = core.OpDesc()
        desc.set_type("clip")
        desc.set_input("X", grad_name)
        desc.set_output("Out", grad_name)
        desc.set_attr("min", self.min)
        desc.set_attr("max", self.max)
        return desc
```

The `BaseErrorClipAttr` have two main member functions:

- **`create_clip_op_desc(self, grad_name)`**

> This function is used to create a C++ `OpDesc` object of `clip_op` and return its pointer to Python. For different error clips require different `clip_op`, the function is defined as virtual in the base class. All derived classes must implement their own versions of this function.

- **`prepend_clip_op_desc(self, op_descs)`**

> This function takes a list of C++ `OpDesc` as input. It checks each `OpDesc` in the list, creates `clip_op`s for every gradient outputs and then appends them to the input list. The input `op_descs` is supposed to be the backward of a certain forward op. It can contain one or more `OpDesc`s (Some op's backward is a combination of several other ops). 

This two functions take effort during the backward building. Just as we showed in the *Usage* section, `Optimizer`'s `minimize` function can take an object of `ErrorClipByValue`(or some other `BaseErrorClipAttr`'s derived class). Inside the `minimize` function, the `prepend_clip_op_desc` function will be send to backward building process as an callback function:

```python
params_grads = append_backward(loss=loss, 
                               parameter_list=parameter_list,
                               no_grad_set=no_grad_set,
                               callback=error_clip.prepend_clip_op_desc)
```

Each time we get the backward of a forward op, we invoke the callback function to append `clip_op` for all the new generated gradients(In the `_append_backward_ops_` function of *backward.py*):

```python
grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
            op.desc, no_grad_dict[block.idx], grad_sub_block_list)
if callback is not None:
    grad_op_desc = callback(grad_op_desc)
```
