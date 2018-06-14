# Kernel Hint Design

## Problem
In PaddlePaddle's [Design](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/execution/switch.md), one Operator may have multiple kernels. Users may have some personal preference to choose a certain type of kernel for an operator, such as `force_cpu` to choose a CPU kernel, `use_cudnn` to choose a CUDNN kernel, we need to provide a way for users to do this.

In the current design, we use KernelType to describe one kernel.

```cpp
struct KernelType {
  Place place_;
  DataType data_type_;
  LayoutType layout_;
};
```
 `place_` `data_type_` and `layout_` can be got from the input tensors of the operator, `GetActualKernelType(inputs)` use inputs to infer the proper kernel key that fit the incoming data, but users can not directly configure it.

The [design](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/execution/switch.md) also provides a virtual method `GetExpectedKernelType` that user can overload and use to choose the KernelType they want to use.

So we should send the information user defined in proto to `GetExpectedKernelType` for choosing a kernel.

The problem is, how should we define and send the information for `GetExpectedKernelType` to use?

## Solution

### Potential choice
1. Do nothing, let the user add the information they want to operator‘s attribute and get them inside `GetExpectedKernelType`, this can work properly. But there is a little problem that users may define many kinds of hints for the same purpose, such as `force_cpu`, `use_cpu`, `cpu_kernel` to choose CPU kernel, and `use_cudnn`, `force_cudnn`, `cudnn_kernel` to choose CUDNN kernel.

2. Pre-define all the needed option and use a single attr key such as `kernel_hint` for the user, this is not so flexible if the user wants to define some more kind of hint.

### Final choice
To provide enough flexibility while avoiding confusion definition, we can define some global constants for these attribute names, such as `force_cpu`, `use_cudnn`, `use_mkldnn` for a user to choose.

In C++

```cpp
const std::string kForceCPU = "force_cpu";
const std::string kUseCUDNN = "use_cudnn";
const std::string kUseMKLDNN = "use_mkldnn";

KernelType GetExpectedKernelType() {
  if (Attr<bool>(kForceCPU)) {
    return KernelType(CPUPlace, ...)
  } else {
    ...
  }
}
```

In Python code

```python
FORCE_CPU = core.kForceCPU()

def xx_layer(..., force_cpu=false):
  layer_helper = LayerHelper(...)
  layer_helper.append_op(
    type="xx",
    attr={FORCE_CPU: force_cpu})
```
