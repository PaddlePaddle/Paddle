/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/jit_kernel.h"
#include <string>

#ifdef PADDLE_WITH_MKLML
#include "paddle/fluid/platform/dynload/mklml.h"
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {

namespace jit = platform::jit;

#define SEARCH_BLOCK(src, t, isa)                             \
  if (d < AVX_FLOAT_BLOCK) {                                  \
    Compute = src<t, isa, kLT8>;                              \
  } else if (d == AVX_FLOAT_BLOCK) {                          \
    Compute = src<t, isa, kEQ8>;                              \
  } else if (d > AVX_FLOAT_BLOCK && d < AVX512_FLOAT_BLOCK) { \
    Compute = src<t, isa, kGT8LT16>;                          \
  } else if (d == AVX512_FLOAT_BLOCK) {                       \
    Compute = src<t, isa, kEQ16>;                             \
  } else {                                                    \
    Compute = src<t, isa, kGT16>;                             \
  }

#define SEARCH_ISA_BLOCK(src, t)        \
  if (jit::MayIUse(jit::avx512f)) {     \
    SEARCH_BLOCK(src, t, jit::avx512f); \
  } else if (jit::MayIUse(jit::avx2)) { \
    SEARCH_BLOCK(src, t, jit::avx2);    \
  } else if (jit::MayIUse(jit::avx)) {  \
    SEARCH_BLOCK(src, t, jit::avx);     \
  } else {                              \
    SEARCH_BLOCK(src, t, jit::isa_any); \
  }

// do not include lt8, eq8, eq16
#define FOR_EACH_COMMON_BLOCK(macro_, isa) \
  macro_(isa, kGT8LT16) macro_(isa, kGT16)

#define FOR_EACH_ISA_COMMON_BLOCK(macro_)     \
  FOR_EACH_COMMON_BLOCK(macro_, jit::avx512f) \
  FOR_EACH_COMMON_BLOCK(macro_, jit::avx2)    \
  FOR_EACH_COMMON_BLOCK(macro_, jit::avx)     \
  FOR_EACH_COMMON_BLOCK(macro_, jit::any)

#define FOR_EACH_ALL_BLOCK(macro_, isa)                                        \
  macro_(isa, kLT8) macro_(isa, kEQ8) macro_(isa, kGT8LT16) macro_(isa, kEQ16) \
      macro_(isa, kGT16)

#define FOR_EACH_ISA_ALL_BLOCK(macro_)     \
  FOR_EACH_ALL_BLOCK(macro_, jit::avx512f) \
  FOR_EACH_ALL_BLOCK(macro_, jit::avx2)    \
  FOR_EACH_ALL_BLOCK(macro_, jit::avx)     \
  FOR_EACH_ALL_BLOCK(macro_, jit::any)

#define BIND_KERNEL_WITH_DTYPE(ker_class, ker_func, ker_dtype) \
  template <>                                                  \
  ker_class<ker_dtype>::ker_class(int d) {                     \
    SEARCH_ISA_BLOCK(ker_func, ker_dtype);                     \
  }

#define BIND_KERNEL(ker_class, ker_func)              \
  BIND_KERNEL_WITH_DTYPE(ker_class, ker_func, float); \
  BIND_KERNEL_WITH_DTYPE(ker_class, ker_func, double)

/* VMUL JitKernel */
template <typename T, platform::jit::cpu_isa_t isa, jit_block>
static void VMulCompute(const int n, const T* x, const T* y, T* z) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] * y[i];
  }
}

#ifdef PADDLE_USE_MKLML
#define VMUL_MKL_FLOAT(isa, block)                                 \
  template <>                                                      \
  void VMulCompute<float, isa, block>(const int n, const float* x, \
                                      const float* y, float* z) {  \
    platform::dynload::vsMul(n, x, y, z);                          \
  }

#define VMUL_MKL_DOUBLE(isa, block)                                  \
  template <>                                                        \
  void VMulCompute<double, isa, block>(const int n, const double* x, \
                                       const double* y, float* z) {  \
    platform::dynload::vdMul(n, x, y, z);                            \
  }

FOR_EACH_ISA_COMMON_BLOCK(VMUL_MKL_FLOAT)
FOR_EACH_ISA_ALL_BLOCK(VMUL_MKL_DOUBLE)
#endif

/// lt8
#ifdef PADDLE_USE_MKLML
VMUL_MKL_FLOAT(jit::avx, kLT8)
VMUL_MKL_FLOAT(jit::avx2, kLT8)
VMUL_MKL_FLOAT(jit::avx512f, kLT8)
#endif

/// eq8
#define VMUL_INTRI8_FLOAT(isa)                                    \
  template <>                                                     \
  void VMulCompute<float, isa, kEQ8>(const int n, const float* x, \
                                     const float* y, float* z) {  \
    __m256 tmpx, tmpy;                                            \
    tmpx = _mm256_loadu_ps(x);                                    \
    tmpy = _mm256_loadu_ps(y);                                    \
    tmpx = _mm256_mul_ps(tmpx, tmpy);                             \
    _mm256_storeu_ps(z, tmpx);                                    \
  }

// mkl > avx > for, ">" means better
#ifdef PADDLE_USE_MKLML
VMUL_MKL_FLOAT(jit::avx, kEQ8)
#elif defined __AVX__
VMUL_INTRI8_FLOAT(jit::avx)
#endif
// avx2 > mkl > for
#ifdef __AVX2__
VMUL_INTRI8_FLOAT(jit::avx2)
#elif defined PADDLE_USE_MKLML
VMUL_MKL_FLOAT(jit::avx2, kEQ8)
#endif
// TODO(TJ): test and complete avx512

/// eq16
#ifdef PADDLE_USE_MKLML
// TODO(TJ): test and complete me
VMUL_MKL_FLOAT(jit::avx, kEQ16)
VMUL_MKL_FLOAT(jit::avx2, kEQ16)
VMUL_MKL_FLOAT(jit::avx512f, kEQ16)
#endif

#undef VMUL_INTRI8_FLOAT
#undef VMUL_MKL_FLOAT
#undef VMUL_MKL_DOUBLE

/* VADD */
template <typename T, platform::jit::cpu_isa_t isa, jit_block>
static void VAddCompute(const int n, const T* x, const T* y, T* z) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}

#ifdef PADDLE_USE_MKLML
#define VADD_MKL_FLOAT(isa, block)                                 \
  template <>                                                      \
  void VAddCompute<float, isa, block>(const int n, const float* x, \
                                      const float* y, float* z) {  \
    platform::dynload::vsAdd(n, x, y, z);                          \
  }

#define VADD_MKL_DOUBLE(isa, block)                                  \
  template <>                                                        \
  void VAddCompute<double, isa, block>(const int n, const double* x, \
                                       const double* y, float* z) {  \
    platform::dynload::vdAdd(n, x, y, z);                            \
  }

FOR_EACH_ISA_COMMON_BLOCK(VADD_MKL_FLOAT)
FOR_EACH_ISA_ALL_BLOCK(VADD_MKL_DOUBLE)
#endif

/// lt8
#ifdef PADDLE_USE_MKLML
VADD_MKL_FLOAT(jit::avx, kLT8)
VADD_MKL_FLOAT(jit::avx2, kLT8)
VADD_MKL_FLOAT(jit::avx512f, kLT8)
#endif

/// eq8
#define VADD_INTRI8_FLOAT(isa)                                    \
  template <>                                                     \
  void VAddCompute<float, isa, kEQ8>(const int n, const float* x, \
                                     const float* y, float* z) {  \
    __m256 tmpx, tmpy;                                            \
    tmpx = _mm256_loadu_ps(x);                                    \
    tmpy = _mm256_loadu_ps(y);                                    \
    tmpx = _mm256_add_ps(tmpx, tmpy);                             \
    _mm256_storeu_ps(z, tmpx);                                    \
  }

// mkl > avx > for, ">" means better
#ifdef PADDLE_USE_MKLML
VADD_MKL_FLOAT(jit::avx, kEQ8)
#elif defined __AVX__
VADD_INTRI8_FLOAT(jit::avx)
#endif
// avx2 > mkl > for
#ifdef __AVX2__
VADD_INTRI8_FLOAT(jit::avx2)
#elif defined PADDLE_USE_MKLML
VADD_MKL_FLOAT(jit::avx2, kEQ8)
#endif
// TODO(TJ): test and complete avx512

/// eq16
#ifdef PADDLE_USE_MKLML
// TODO(TJ): test and complete me
VADD_MKL_FLOAT(jit::avx, kEQ16)
VADD_MKL_FLOAT(jit::avx2, kEQ16)
VADD_MKL_FLOAT(jit::avx512f, kEQ16)
#endif

#undef VADD_INTRI8_FLOAT
#undef VADD_MKL_FLOAT
#undef VADD_MKL_DOUBLE

BIND_KERNEL(VMulKernel, VMulCompute);
BIND_KERNEL(VAddKernel, VAddCompute);

#undef BIND_KERNEL
#undef BIND_KERNEL_WITH_DTYPE
#undef FOR_EACH_ISA_ALL_BLOCK
#undef FOR_EACH_ALL_BLOCK
#undef FOR_EACH_ISA_COMMON_BLOCK
#undef FOR_EACH_COMMON_BLOCK
#undef SEARCH_ISA_BLOCK
#undef SEARCH_BLOCK

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
