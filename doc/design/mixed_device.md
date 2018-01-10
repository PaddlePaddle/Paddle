## Design Doc: mixed device support

### Problem
A Paddle program is consturcted by many Operators, many of them have kernels that can run on different devices. Most of the time, all operators will run on the same device, but in some condition, they need to run on different devices, such as:

1. one operator only has CPU kernel, but other operators can run CUDA place, we need to support other operators run on CUDA but this run on CPU.
2. Some parameter is too large to put into GPU memory, like embedding table. It need to be put on Host memory, and operatores that use this parameter need to run on CPU, like lookup_table_op.

The input and output is on the same place with the operators. If two related operators OP1 and OP2 are on the same place, the output `op1_2_op2` of OP1 can be used directly by OP2.

```
OP1(CPUPlace)
     |
 op1_2_op2
     |
OP2(CPUPlace)
```

But If OP1 and OP2 are on different devices, the `op1_2_op2` can not be used directly be OP2. The framework should be able to distinguish this kind of situation and do the data transform automatically.


```
OP1(CPUPlace)
     |
 op1_2_op2(on cpu)
     |
[transform](from cpu to gpu)
     |
 op1_2_op2(on gpu)
     |
OP2(CUDAPlace)
```


### Sulotion
We use [`OpKernelType `](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/op_kernel_type.h) to describe a tensor and a kernel that can use this tensor:

```cpp
struct OpKernelType {
  proto::DataType data_type_;
  DataLayout data_layout_;
  platform::Place place_;
  LibraryType library_type_;
}
```

#### Steps to do auto device transform for data:

1. find out expected_kernel that will be used this time.
1. find out the actual OpKernelType info for each input. If one input tensor have different place with the expected_kernel then:
1. do data transform for this input. the transformed input will be put into a subscope of current running scope with the same name, so Op2 will get the transformed data with the same name. A Pseudo code is:

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
      auto* out = DataDeviceTransform(expected_kernel_key,
                                      kernel_type_for_var,
                                      *tensor_in);
      CopyVariableWithTensor(...);
    }
  }

  auto kernel = kernels.find(expected_kernel_key);

  kernel->Compute(ExecutionContext(...));
}
```
