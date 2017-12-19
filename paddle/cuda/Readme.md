## Single release for PaddlePaddle CPU Image 

### Background

Currently, PaddlePaddle supports AVX and SSE3 intrinsics (extensions to the x86 instruction set architecture). When using CMake to compile PaddlePaddle source code, it will check and detect the host which SIMD instruction is supported, then automatically set the legal one.  Developer or user also could manually set CMake option `WITH_AVX=ON/OFF` before PaddlePaddle compilation. That's good for local usage.


### Problem Involved

Nonetheless, from the perspective of the deployment, there are some drawbacks:

1. The online runtime environment is very complex, if an older node does not support AVX or others,
PaddlePaddle will crash and throw out `illegal instruction is used`. This problem will appear
frequently on cluster environment, like Kubernetes. **It must be addressed before PaddlePaddle on Cloud**

2. Once new version is ready to deliver, we have to release more products to users, for example, `no-avx-cpu`, `avx-cpu`, `no-avx-gpu`, `avx-gpu`. Users do not need to care about details. It sucks!


### How to Address it?

To address this issue, there are three primary components:

1. [Done] Runtime Check:

        We can utilize CPU ID information to check SIMD info at runtime. This functionality already merged into
        current develop branch. For full details, please check out [CpuId.cpp](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/utils/CpuId.cpp) and [CpuId.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/utils/CpuId.h).


2. [Pending] Adjust `cuda` Directory.

        Since the current `cuda` directory includes heterogeneous source code (cpu and gpu), we want to refactor `cuda` directory. For simplicity, different simd intrinsics will be inside the different directories. we need to
        modified CMake files to support this solution.

3. [Pending] Modify CMake files.

        Different simd intrinsics will be inside the different directories. we need to modified CMake files to support this solution. Each directory uses the different compile options (`-mavx` or `-msse`) to generate the corresponding binaries. Then, at runtime, using SIMD flags `HAS_AVX`, `HAS_SSE` automatically detect and select the supported branch (intrinsics) to execute.


### Conclusion

The method could fix the releases and deployment problems.
