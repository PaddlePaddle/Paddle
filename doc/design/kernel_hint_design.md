## Problem
In PaddlePaddle's [Design](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/switch_kernel.md), one Operator may have multiple kernels. Users may have some personal preference to choose a certain type of kernel for an operator, such as `force_cpu` to use a CPU kernel, `use_cudnn` to choose a CUDNN kernel, we need to provide a way for a user to do this.

In the current design,  we use KernelType to describe one kernel.

```cpp
struct KernelType {
  Place place_;
  DataType data_type_;
  LayoutType layout_;
};
```
 `place_` `data_type_` and `layout_` can come from the input tensor of the operator, `GetActualKernelType(inputs)` use inputs to infer the proper kernel key that fit the incoming data, user can not config it.

The design also provides a virtual method `GetExpectedKernelType` that user can overload and choose the KernelType they want to use.

so, we should send the information user defined in proto to `GetExpectedKernelType` for choosing a kernel.

The problem is, how should we define and send the information for `GetExpectedKernelType` to use?

## Solution
1, Do nothing, let the user add the information they want to operator‘s attribute and get them inside `GetExpectedKernelType`, this can work right. But there is a little problem that users may define many kinds of hints for the same purpose, such as `force_cpu`, `use_cpu`, `CPU` for CPU kernel, and `use_cudnn`, `force_cudnn`, `cudnn_kernel` for use of CUDNN kernel.

2, Pre-define all the needed option and use a single attr key such as `kernel_hint` for the user, this is not so flexible if the user wants to define some more kind of hint.


To provide enough flexibility while avoiding confusion definition, we can predefine some options, such as `force_cpu`, `use_cudnn`, `use_mkldnn` for a user to choose.

```cpp
const std::string kNonHint = "";
const std::string kForceCPU = "force_cpu";
const std::string kUseCUDNN = "use_cudnn";
const std::string kUseMKLDNN = "use_mkldnn";

KernelType GetExpectedKernelTyp() {
    // "kernel_hint" is a user defined attribute name
	if (Attr<std::string>("kernel_hint") == kForceCPU) {
		return KernelType(CPUPlace, ...)
	} else {
		...
	}
}
```

In Python code

```python
def xx_layer(..., kernel_hint=None):
	layer_helper = ...
	layer_helper .append_op(
		type="xx",
		# "kernel_hint" should be the same with the attr name in CPP
		attr={"kernel_hint": kernel_hint or ""})
```
