# JIT Kernel

JIT(Just In Time) Kernel contains actually generated code and some other implemenations with the same logic.
Each implementation has its own condition to use, defined in `CanBeUsed`.
They are combined together to get the best performance of one single independent function.
They could be some very simple functions like vector multiply, or some complicated functions like LSTM.
And they can be composed with some other exited jit kernels to build up a complex function.
Currently it's only supported on CPU yet.

## Contents

```txt
PaddlePaddle/Paddle/paddle/fluid/
├── ...
└── operators/
    ├── .../
    └── jit/
        ├── ...
        ├── gen/
        │   └── ...
        |── more/
        │   ├── ...
        │   ├── mkl/
        │   │   └── ...
        │   ├── mkldnn/
        │   │   └── ...
        │   ├── mix/
        │   │   └── ...
        │   ├── intrinsic/
        │   │   └── ...
        │   └── openblas/
        │       └── ...
        └── refer/
            └── ...
```

All basical definations of jit kernels are addressed in `paddle/fluid/operators/jit` including these three key folders `refer`, `gen`, `more`. There is only one unique name for each kernel while may have seraval implementations with same functionality.

- `refer`: Each kernel must have one reference implementation on CPU, and it should only focus on the correctness and should not depends on any third-party libraries.
- `gen`: The code generated should be kept here. They should be designed focusing on the best performance, which depends on Xbyak.
- `more`: All other implementations should be kept in this folder with one directory corresponding to one library kind or method kind, such as mkl, mkldnn, openblas or intrinsic code. Each implementation should have it advantage.

## How to use

We present these methods to get the functions:
- `GetAllCandidateFuncs`. It can return all the implementations supported. All of the implementations can get the same result. You can do some runtime benchmark to choose which should actually be used.
- `GetDefaultBestFunc`. It only return one default function pointer, which is tuning offline with some genenal configures and attributes. This should cover most situations.
- `KernelFuncs::Cache()`. It can get the default functions and save it for next time with the same attribute.
- `GetReferFunc`. It can only get the reference code in CPU, and all the others implementations have same logic with this reference code.

And here are some examples:

Get from cache:

```cpp
    using T = float;
    jit::seq_pool_attr_t attr(width, jit::SeqPoolType::kSum);
    auto seqpool_func = jit::KernelFuncs<jit::SeqPoolTuple<T>, platform::CPUPlace>::Cache().At(attr);
    seqpool_func(src_data, dst_data, &attr);
```

Get all implementations and run once:

```cpp
    using T = float;
    jit::seq_pool_attr_t attr(width, jit::SeqPoolType::kSum);
    auto funcs = jit::GetAllCandidateFuncsWithTypes<jit::SeqPoolTuple<T>, platform::CPUPlace>(attr);
    for (auto f : funcs) {
        LOG(INFO) << "Kernel implementation type: " << f.first;
        f.second(src_data, dst_data, &attr);
    }
```

All kernels are inlcuded in `paddle/fluid/operators/jit/kernels.h`, which is automatically generated in compile time, you can only include this one header to get all the registered kernels.

## Solid Test

- Unit Test
    All functions should be compared with the corresponding reference functions, including data tyep `float` and `double`.
- Benchmark
    All functions should be tested, and make sure the `jit::GetDefaultBestFunc` function obtain the best performance with all attributes.

# How to add new kernel

## Required

1. Add `your_key` at `KernelType`.
2. Add your new `KernelTuple` which must include `your_key`. It should be a combination of the data type, attribute type and function type. You can refer `SeqPoolTuple`.
3. Add reference function of `your_key`.
Note:
    - this should be run on CPU and do not depend on any third-party.
    - Add `USE_JITKERNEL_REFER(your_key)` in `refer/CmakeLists.txt` to make sure this code can be used.
4. Add unit test in `test.cc`, and verfiy at least `float` and `double`.
Test more data type for some special functions if necessary, for example `int8`.
5. Add functions in `benchmark.cc` to test all function of same `KernelType`. Make sure `GetDefaultBestFunc` always get the best one.

## Optional

Add more implementations of `your_kery` for performance enhancement.

1. Add functions based on generated code in `gen`. It should be derived from `JitCode` and should have correpsonding creator from `JitCodeCreator` which will be registered on the `your_key`.
2. If new attribute type is added, you should specialize `JitCodeKey` of this type.
3. Add more functions in `more`，you can use any third party you wish, like mkl, mkldnn or intrinsic code to reach the best performance.
