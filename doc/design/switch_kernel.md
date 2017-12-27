## Background
Every operator has many kernels because there are multiple data types, places, data layout that Fluid supports. We use the `KernelType` to describe kernel types that operators can hold. 

The `KernelType` is as follows.

```
struct KernelType {
  Place place_;
  DataType data_type_;
  LayoutType layout_;
};
```

The `place_` is a descriptor of the device and the computational library, e.g., `MKLDNNPlace`, `CUDAPlace`.

The `data_type_` is the data type that this kernel performs on, e.g., `FP32`, `INT64`. Note that one kernel may have inputs with different data types. However, it will be a major `data_type`. For example, the `cross_entropy` takes `int64` as it label, and `double`/`float` as its input logit and output cost. The major `data_type` of `cross_entropy` is `float`/`double`.

The `layout` is useful for some computational library. One example is that MKLDNN uses many kinds of layout, such as `nChw8c`. Each kind of layout will invoke the different kernel.

## Problem

We register a kernel for every operator and every kernel type ideally. However, it is impracticable for the following situations.

1. Some operators, like CRF, are complicated and inefficient to be implemented on GPU. The CRF operator will only have a CPU kernel.
2. Some operators will take too many memory. It is better to force them into CPU. However, the rest of operators in this neural network will be performed on GPU, i.e., model parallel problem.
3. Some layout and place are particular. One example is that MKLDNN uses `nChw8` and there is no other library uses `nChw8c`.

Problems under these situations are similar. We can formalise this problem as follow.

We register kernels with types $KT = \{kt_1, kt_2, kt_3, ...\}$ for one operator. The inputs of this operator should be run on kernel type $kt_{?}$, which the $kt_{?} \notin KT$. How to cast the input of this operator from $kt_{?}$ to any of kernel type in $KT$.

## Solution

It is clearly that transforming inputs of an operator toadapt another kernel type is not related to the particular operator. So we should register these transformation methods as global methods.

We can infer a kernel type from the inputs of an operators. We let this kernel type as `actual kernel type`, which means this kernel type is the actually kernel type that operator should be performed.

We can get a kernel type by 1) The configuration of operator description. (Users may want to force use `MKL` for `conv` operator). 2) The place of the current executor. (Executor is running on GPU). This kernel type is what we expect the operator will be performed on. We let this kernel type as `expect kernel type`.

We transform the input data from `actual` to `expect` if the expect kernel type is not as same as actual kernel type.

The algorithm is described as follow

```cpp
using DataTransformationFN = std::function<void(const Tensor& in, Tensor* out)>;
using KernelTypePair = std::pair<KernelType, KernelType>;

map<KernelTypePair, DataTransformationFN> g_data_transformation_;

void OpWithKernel::Run() {
  vec<Tensor> inputs = ...
  auto actual_kernel_type = GetActualKernelType(inputs);
  
  // The expected kernel type is related to actual kernel type.
  // For the most operators, the expected kernel type is as same as
  // actual kernel type.
  //
  // So we pass `actual_kernel_type` as a parameter of 
  // GetExpectedKernelType
  auto expect_kernel_type = GetExpectedKernelType(actual_kernel_type);
  
  auto trans = g_data_transformation_[{actual_kernel_type, expect_kernel_type}];
  
  kernel.run(trans(inputs));
}
```
