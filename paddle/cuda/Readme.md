## Runtime Check SIMD for x86 architecture

### Background

Currently, PaddlePaddle supports AVX and SSE3 intrinsics (extensions to the x86 instruction set architecture). When using CMake to compile PaddlePaddle source code, it will check and detect the host which SIMD instruction is supported, then automatically set the legal one.  Developer or user also could manually set CMake option `WITH_AVX=ON/OFF` before PaddlePaddle compilation. That's good for local usage.


### Problem Involved

Nonetheless, from the perspective of the deployment, there are some drawbacks:

1. The online runtime environment is very complex, if an older node does not support AVX or others,
PaddlePaddle will crash and throw out `illegal instruction is used`. This problem will appear
frequently on cluster environment, like Kubernetes. **It must be addressed before PaddlePaddle on Cloud**

2. Once new version is ready to deliver, we have to release more products to users, for example, `no-avx-cpu`, `avx-cpu`, `no-avx-gpu`, `avx=gpu`. Users do not need to care about details. It sucks!


### How to Address it?

1. We can utilize CPU ID information to check SIMD info at runtime. This functionality already merged into
current develop branch. For full details, please check out [CpuId.cpp](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/utils/CpuId.cpp) and [CpuId.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/utils/CpuId.h).

You can use `HAS_SIMD(__flags)` to runtime check SIMD. For instance,

```c++
 if (HAS_SIMD(SIMD_AVX2 | SIMD_FMA4)) {
      avx2_fm4_stub();
 } else if (HAS_SIMD(SIMD_SSE3)) {
      sse3_stub();
 }
```

`avx2_fm4_stub` and `sse3_stub` could be located in different directory:

```text
------x84---naive
       |     |
       |     |---avx2 -- avx2_fm4_stub()
       |     |
       |     |---sse  -- sse_stub()
       |     |
       |     |---sse3 -- sse3_stub()
       |
       arm--- ...
```

Here, each directory uses the different compile options (`-mavx` or `-msse`) to generate the corresponding binaries. Then, at
runtime, it could be `if(HAS_SIMD(__flags)` can select the supported branch (intrinsics) to execute.

The method could fix the releases and deployment problems.


### How to implement it?

Since the current `cuda` directory includes heterogeneous source code, we want to refactor `cuda` directory as follows:

```
kernels--- cpu --- inc -- x86 -- avx ----- avx_mathfun.h activation.h gru.h ...
        |       |             |
        |       |             |- naive --- activation.h gru.h ...
        |       |
        |       |- src -- x86 -- avx ----- activation.cc
        |                    |         |- gru.cc
        |                    |         |- ...
        |                    |
        |                    |- naive --- activation.cc
        |                    |         |- gru.cc
        |                              |- ...
        |- gpu -- ...
```

For simplicity, different arches or intrinsics will be inside the different directories. we need to
modified CMake files to support this solution.


### Reference

AVX Cheat Sheet, TUM, https://db.in.tum.de/~finis/x86%20intrinsics%20cheat%20sheet%20v1.0.pdf
