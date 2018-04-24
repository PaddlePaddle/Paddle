# Kernel Selection

## Background
Every operator has many kernels because there are multiple data types, places, data layout, library type that Fluid supports. We use the `OpKernelType ` to describe kernel types that operators can hold.

The `OpKernelType ` is as follows:

```cpp
struct OpKernelType {
  Place place_;
  DataType data_type_;
  DataLayout data_layout_;
  LibraryType library_type_;
};
```

- The `place_` is a descriptor of the device, e.g., CPUPlace, CUDAPlace.

- The `data_type_` is the data type that this kernel performs on, e.g., `FP32`, `INT64`. Note that one kernel may have inputs with different data types. However, it will be a major `data_type`. For example, the `cross_entropy` takes `int64` as it label, and `double`/`float` as its input logit and output cost. The major `data_type` of `cross_entropy` is `float` or `double`.

- The `data_layout_ ` is useful for some computational library. One example is that MKLDNN uses many kinds of layout, such as `nChw8c`. Each kind of layout will invoke the different kernel.

- The `library_type_` describes the computational library, e.g., `MKLDNN`, `CUDNN`.

## Problem

We register a kernel for every operator and every kernel type ideally. However, it is impracticable for the following situations.

1. Some operators, like CRF, are complicated and inefficient to be implemented on GPU. The CRF operator will only have a CPU kernel.
2. Some operators will take too many memory. It is better to force them into CPU. However, the rest of operators in this neural network will be performed on GPU, i.e., model parallel problem.
3. Some layout and place are particular. One example is that MKLDNN uses `nChw8` and there is no other library uses `nChw8c`.

Take one situation to give a detailed explanation, if we have two Operators: OP1 and OP2, OP1 has one output `op1_to_op2`, and `op1_to_op2` is the input of OP2.

If OP1 and OP2 run on the same place(for example CPUPlace), then `op1_2_op2` can be used directly by OP2.

```
OP1(CPUPlace)
     |
 op1_2_op2
     |
OP2(CPUPlace)
```

If OP1 and OP2 run one different place, then OP2 cannot `use op1_2_op2` directly.

Problems under these situations are similar. We can formalize this problem as follow.

We register kernels with types $KT = \{kt_1, kt_2, kt_3, ...\}$ for one operator. The inputs of this operator should be run on kernel type $kt_{?}$, which the $kt_{?} \notin KT$. How to cast the input of this operator from $kt_{?}$ to any of kernel type in $KT$.

## Solution: data transform

It is clear that transforming inputs of an operator to adapt another kernel type is not related to the particular operator. So we should register these transformation methods as global methods.

We can infer kernel type for each input of an operator. We let this kernel type as `actual kernel type for var`, which means this kernel type is the kernel type that can process this input variable.

We can get a kernel type by 1) The configuration of operator description. (Users may want to force use `MKL` for `conv` operator). 2) The place of the current executor. (Executor is running on GPU). This kernel type is what we expect the operator will be performed on. We let this kernel type as `expect kernel type`.

We transform the input data from `actual` to `expect` if the actual kernel type is not as same as expect kernel type.

The algorithm is described as following

```cpp
void OperatorWithKernel::Run(
        const Scope& scope,
        const platform::Place& place) const {
  ExecutionContext ctx(...);
  auto expected_kernel_key = this->GetExpectedKernelType(ctx);

  Scope& new_scope = scope.NewScope();

  for (auto& var_name : this->Inputs()) {
    auto* tensor_in = GetTensor(var_name);
    auto kernel_type_for_var = this->GetKernelTypeForVar(...);
    if (kernel_type_for_var.place_ != expected_kernel_key.place_) {
      auto* trans_var = new_scope.Var(var_name);
      auto* out = DataTransform(expected_kernel_key,
                                kernel_type_for_var,
                                *tensor_in);
      CopyVariableWithTensor(...);
    }
  }

  auto kernel = kernels.find(expected_kernel_key);
  kernel->Compute(ExecutionContext(...));
}
```

then the actual process for the multi-device above will be:

```
OP1(CPUPlace)
     |
op1_2_op2(on CPU)
     |
[transform](from CPU to GPU)
     |
op1_2_op2(on GPU)
     |
OP2(CUDAPlace)
```
