# Error Clip

## Overview

Error clip is widely used in model training to prevent gradient exploding. It takes some specific rules to adjust variables' gradients and prevent them from being too large. With it, values of a gradient will be checked before they are taken by the next `grad_op` and be shrunk if necessary.
## Usage

Users are allowed to assign different error clip methods or attributes to different `Variable`s. Users can specify it as a parameter of `Variable`'s constructor:

```python
var = framework.Variable(..., error_clip=myErrorClip, ...)
```

The default value of `error_clip` is `None`, which means no error clip is employed. When it's not `None`, it should take an object of `BaseErrorClipAttr`'s derived class. So far, `BaseErrorClipAttr` has only one derived class: `ErrorClipByValue`, whose constructor is:

```python
ErrorClipByValue(max, min=None)
```

`max` and `min` represent the maximal and minimal clip threshold respectively. In backward pass, all values of `var`'s gradient greater than `max` or less than `min` will be clipped to `max` and `min` respectively. When the `min` is None, the minimal threshold will be assigned with `-max` automatically.

So we can enable the error clip with threshold `[-5.0, 5.0]` for variable `var` by:

```python
var = framework.Variable(..., error_clip=ErrorClipByValue(max=5.0), ...)
```

## Implementation

The `BaseErrorClipAttr` and its derived class `ErrorClipByValue` are defined in *clip.py*.

```python
class BaseErrorClipAttr(object):
    def append_clip_op(self, block, grad_name):
        raise NotImplementedError()


class ErrorClipByValue(BaseErrorClipAttr):
    def __init__(self, max, min=None):
        max = float(max)
        if min is None:
            min = -max
        else:
            min = float(min)
        self.max = max
        self.min = min

    def append_clip_op(self, block, grad_name):
        clip_op_desc = block.desc.append_op()
        clip_op_desc.set_type("clip")
        clip_op_desc.set_input("X", [grad_name])
        clip_op_desc.set_output("Out", [grad_name])
        clip_op_desc.set_attr("min", self.min)
        clip_op_desc.set_attr("max", self.max)
```

The `BaseErrorClipAttr` have one main member functions: `append_clip_op(self, block, grad_name)`.

This function is used to create a `clip_op` and append it to the end of given `block`. For different error clip algorithm require different `clip_op`, the function is defined as virtual in the base class. All derived classes must implement their own versions of this function.

These `clip_op`s should be inserted after `grad_op`s whose output gradients need to be clipped. It is equivalent to appending some `clip_op`s to the end of the target block every time a new `grad_op` is added.

```python
for op_desc in grad_op_descs:
        new_op_desc = target_block.desc.append_op()
        new_op_desc.copy_from(op_desc)
        callback(block=target_block, context=grad_to_var)
```

Here we employ a callback function to complete this kind of jobs. In `_append_backward_ops_` function, each time after a `grad_op` is added to the `target_block`, a callback function is invoked. The logic of `clip_op` appending can be implemented inside the callback function.

The callback function for `clip_op` appending is defined in *clip.py*:

```python
def error_clip_callback(block, context):
    # the context is a grad_to_var map
    grad_to_var = context
    op_desc = block.desc.op(block.desc.op_size() - 1)
    for grad_n in filter(lambda n: grad_to_var.has_key(n),
                         op_desc.output_arg_names()):
        fwd_var = block.var_recursive(grad_to_var[grad_n])
        error_clip = getattr(fwd_var, "error_clip", None)
        if not (error_clip is None or isinstance(error_clip,
                                                 BaseErrorClipAttr)):
            raise TypeError(
                "Variable's error_clip should be an instance of BaseErrorClipAttr or None."
            )
        if error_clip is not None:
            error_clip.append_clip_op(block, grad_n)
```

This function takes a `block` and a `context`(which is actually a grad\_to\_var map) as inputs. It checks each output of the last `OpDesc` in the `block`. Notice that the last `OpDesc` of the `block` must be a `grad_op` and its outputs must be some forward variables' gradients. If an output gradient's corresponding forward variable has an attribute of `error_clip`, `error_clip_callback` will call the `error_clip`'s `append_clip_op` function to append the required `clip_op` into the `block`.
