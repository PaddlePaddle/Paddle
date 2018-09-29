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
#include "paddle/fluid/operators/math/jit_kernel_macro.h"
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

/* VMUL JitKernel */
template <typename T, platform::jit::cpu_isa_t isa, jit_block>
class VMulKernelImpl : public VMulKernel<T> {
 public:
  void Compute(const int n, const T* x, const T* y, T* z) const override {
    for (int i = 0; i < n; ++i) {
      z[i] = x[i] * y[i];
    }
  }
};

#ifdef PADDLE_WITH_MKLML
#define MKL_FLOAT(isa, block)                                        \
  template <>                                                        \
  void VMulKernelImpl<float, isa, block>::Compute(                   \
      const int n, const float* x, const float* y, float* z) const { \
    platform::dynload::vsMul(n, x, y, z);                            \
  }

#define MKL_DOUBLE(isa, block)                                          \
  template <>                                                           \
  void VMulKernelImpl<double, isa, block>::Compute(                     \
      const int n, const double* x, const double* y, double* z) const { \
    platform::dynload::vdMul(n, x, y, z);                               \
  }

FOR_EACH_ISA(MKL_FLOAT, kGT16);
FOR_EACH_ISA_BLOCK(MKL_DOUBLE);
#endif

#define INTRI8_FLOAT(isa)                                            \
  template <>                                                        \
  void VMulKernelImpl<float, isa, kEQ8>::Compute(                    \
      const int n, const float* x, const float* y, float* z) const { \
    __m256 tmpx, tmpy;                                               \
    tmpx = _mm256_loadu_ps(x);                                       \
    tmpy = _mm256_loadu_ps(y);                                       \
    tmpx = _mm256_mul_ps(tmpx, tmpy);                                \
    _mm256_storeu_ps(z, tmpx);                                       \
  }

// avx > for > mkl
#ifdef __AVX__
INTRI8_FLOAT(jit::avx);
#endif
#ifdef __AVX2__
INTRI8_FLOAT(jit::avx2);
#endif
#ifdef __AVX512F__
INTRI8_FLOAT(jit::avx512f);
#endif
// TODO(TJ): eq16 test and complete avx512
#undef INTRI8_FLOAT
#undef MKL_FLOAT
#undef MKL_DOUBLE

/* VADD JitKernel */
template <typename T, platform::jit::cpu_isa_t isa, jit_block>
class VAddKernelImpl : public VAddKernel<T> {
 public:
  void Compute(const int n, const T* x, const T* y, T* z) const override {
    for (int i = 0; i < n; ++i) {
      z[i] = x[i] + y[i];
    }
  }
};

#ifdef PADDLE_WITH_MKLML
#define MKL_FLOAT(isa, block)                                        \
  template <>                                                        \
  void VAddKernelImpl<float, isa, block>::Compute(                   \
      const int n, const float* x, const float* y, float* z) const { \
    platform::dynload::vsAdd(n, x, y, z);                            \
  }

#define MKL_DOUBLE(isa, block)                                          \
  template <>                                                           \
  void VAddKernelImpl<double, isa, block>::Compute(                     \
      const int n, const double* x, const double* y, double* z) const { \
    platform::dynload::vdAdd(n, x, y, z);                               \
  }

FOR_EACH_ISA(MKL_FLOAT, kGT16);
FOR_EACH_ISA_BLOCK(MKL_DOUBLE);
#endif

#define INTRI8_FLOAT(isa)                                            \
  template <>                                                        \
  void VAddKernelImpl<float, isa, kEQ8>::Compute(                    \
      const int n, const float* x, const float* y, float* z) const { \
    __m256 tmpx, tmpy;                                               \
    tmpx = _mm256_loadu_ps(x);                                       \
    tmpy = _mm256_loadu_ps(y);                                       \
    tmpx = _mm256_add_ps(tmpx, tmpy);                                \
    _mm256_storeu_ps(z, tmpx);                                       \
  }
#ifdef __AVX__
INTRI8_FLOAT(jit::avx);
#endif
#ifdef __AVX2__
INTRI8_FLOAT(jit::avx2);
#endif
#ifdef __AVX512F__
INTRI8_FLOAT(jit::avx512f);
#endif
// TODO(TJ): eq16 test and complete avx512

#undef INTRI8_FLOAT
#undef MKL_FLOAT
#undef MKL_DOUBLE

/* VSCAL JitKernel */
template <typename T, platform::jit::cpu_isa_t isa, jit_block>
class VScalKernelImpl : public VScalKernel<T> {
 public:
  void Compute(const int n, const T a, const T* x, T* y) const override {
    for (int i = 0; i < n; ++i) {
      y[i] = a * x[i];
    }
  }
  void Compute(const int n, const T a, T* x) const override {
    for (int i = 0; i < n; ++i) {
      x[i] = a * x[i];
    }
  }
};

#ifdef PADDLE_WITH_MKLML
#define MKL_FLOAT(isa, block)                                                  \
  template <>                                                                  \
  void VScalKernelImpl<float, isa, block>::Compute(const int n, const float a, \
                                                   float* x) const {           \
    platform::dynload::cblas_sscal(n, a, x, 1);                                \
  }

#define MKL_DOUBLE(isa, block)                        \
  template <>                                         \
  void VScalKernelImpl<double, isa, block>::Compute(  \
      const int n, const double a, double* x) const { \
    platform::dynload::cblas_dscal(n, a, x, 1);       \
  }

FOR_EACH_ISA(MKL_FLOAT, kGT16);
FOR_EACH_ISA_BLOCK(MKL_DOUBLE);
#endif

#define INTRI8_FLOAT(isa)                                           \
  template <>                                                       \
  void VScalKernelImpl<float, isa, kEQ8>::Compute(                  \
      const int n, const float a, const float* x, float* y) const { \
    __m256 tmp;                                                     \
    __m256 scalar = _mm256_set1_ps(a);                              \
    tmp = _mm256_loadu_ps(x);                                       \
    tmp = _mm256_mul_ps(tmp, scalar);                               \
    _mm256_storeu_ps(y, tmp);                                       \
  }
#define INTRI8_INPLACE_FLOAT(isa)                                             \
  template <>                                                                 \
  void VScalKernelImpl<float, isa, kEQ8>::Compute(const int n, const float a, \
                                                  float* x) const {           \
    __m256 tmp;                                                               \
    __m256 scalar = _mm256_set1_ps(a);                                        \
    tmp = _mm256_loadu_ps(x);                                                 \
    tmp = _mm256_mul_ps(tmp, scalar);                                         \
    _mm256_storeu_ps(x, tmp);                                                 \
  }

#ifdef __AVX__
INTRI8_FLOAT(jit::avx);
INTRI8_INPLACE_FLOAT(jit::avx);
#endif
#ifdef __AVX2__
INTRI8_FLOAT(jit::avx2);
INTRI8_INPLACE_FLOAT(jit::avx2);
#endif
#ifdef __AVX512F__
INTRI8_FLOAT(jit::avx512f);
INTRI8_INPLACE_FLOAT(jit::avx512f);
#endif
// TODO(TJ): eq16 test and complete avx512

#undef INTRI8_FLOAT
#undef INTRI8_INPLACE_FLOAT
#undef MKL_FLOAT
#undef MKL_DOUBLE

REGISTER_JITKERNEL(vmul, VMulKernel);
REGISTER_JITKERNEL(vadd, VAddKernel);
REGISTER_JITKERNEL(vscal, VScalKernel);

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
