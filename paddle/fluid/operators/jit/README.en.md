# JIT Kernel

JIT(Just In Time) Kernel contains actually generated code and some other implemenations with the same logic.
Each implementations has its own condition to use, defined in `UseMe`.
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

One simple function `jit::Get`, which is very easy to use, is supported to get the kernel.
It can automatically return the expected function with best performance under the given attributes. 
All kernels are inlcuded in `paddle/fluid/operators/jit/kernels.h`, you can only include this one header to get all the registered kernels.

## Solid Test

- Unit Test
    All functions should be compared with the corresponding reference functions, including data tyep `float` and `double`.
- Benchmark
    All functions should be tested, and make sure the `jit::Get` function obtain the best performance with all attributes.

# How to add new kernel

## Required

1. Add `your_key` at `KernelType`.
2. Add reference function of `your_key`. 
Note:
    - this should be run on CPU and do not depend on any third-party.
    - Add `USE_JITKERNEL_REFER(your_key)` in `refer/CmakeLists.txt` to make sure this code can be used.
3. Add unit test in `test.cc`, and verfiy at least `float` and `double`.
Test more data type for some special functions if necessary, for example `int8`.
4. Add functions in `benchmark.cc` to test all function of same `KernelType`. Make sure `jit::Get` always get the best one.

## Optional

Add more implementations of `your_kery` for performance enhancement.

1. Add functions based on generated code in `gen`. It should be derived from `JitCode` and should have corepsonding creator from `JitCodeCreator` which will be registered on the `your_key`.
Note: Add new `KernelTuples` if necessary，your can refer to `XYZNTuples`.
Specialie method `JitCodeKey` when add new attribute type。
2. Add more functions in `more`，you can use any third party you wish, like mkl, mkldnn or intrinsic code to reach the best performance.
