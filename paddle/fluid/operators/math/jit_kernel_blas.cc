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

#define NEW_IMPL(src, t, isa, k)         \
  p = std::dynamic_pointer_cast<src<t>>( \
      std::make_shared<src##Impl<t, isa, k>>())

#define SEARCH_BLOCK(src, t, isa)                             \
  if (d < AVX_FLOAT_BLOCK) {                                  \
    NEW_IMPL(src, t, isa, kLT8);                              \
  } else if (d == AVX_FLOAT_BLOCK) {                          \
    NEW_IMPL(src, t, isa, kEQ8);                              \
  } else if (d > AVX_FLOAT_BLOCK && d < AVX512_FLOAT_BLOCK) { \
    NEW_IMPL(src, t, isa, kGT8LT16);                          \
  } else if (d == AVX512_FLOAT_BLOCK) {                       \
    NEW_IMPL(src, t, isa, kEQ16);                             \
  } else {                                                    \
    NEW_IMPL(src, t, isa, kGT16);                             \
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

#define DEFINE_WITH_DTYPE(ker_key, ker_class, ker_dtype, dtype_key)        \
  template <>                                                              \
  const std::shared_ptr<ker_class<ker_dtype>>                              \
  KernelPool::Get<ker_class<ker_dtype>>(int d) {                           \
    std::string key = #ker_key #dtype_key + std::to_string(d);             \
    if (kers_.find(key) == kers_.end()) {                                  \
      std::shared_ptr<ker_class<ker_dtype>> p;                             \
      SEARCH_ISA_BLOCK(ker_class, ker_dtype);                              \
      kers_.insert({key, std::dynamic_pointer_cast<Kernel>(p)});           \
      return p;                                                            \
    }                                                                      \
    return std::dynamic_pointer_cast<ker_class<ker_dtype>>(kers_.at(key)); \
  }

#define REGISTER_BLAS_JITKERNEL(ker_key, ker_class) \
  DEFINE_WITH_DTYPE(ker_key, ker_class, float, f);  \
  DEFINE_WITH_DTYPE(ker_key, ker_class, double, d)

#define FOR_EACH_ISA(macro_, block) \
  macro_(jit::avx512f, block);      \
  macro_(jit::avx2, block);         \
  macro_(jit::avx, block);          \
  macro_(jit::isa_any, block)

#define FOR_EACH_BLOCK(macro_, isa) \
  macro_(isa, kLT8);                \
  macro_(isa, kEQ8);                \
  macro_(isa, kGT8LT16);            \
  macro_(isa, kEQ16);               \
  macro_(isa, kGT16)

#define FOR_EACH_ISA_BLOCK(macro_)      \
  FOR_EACH_BLOCK(macro_, jit::avx512f); \
  FOR_EACH_BLOCK(macro_, jit::avx2);    \
  FOR_EACH_BLOCK(macro_, jit::avx);     \
  FOR_EACH_BLOCK(macro_, jit::isa_any)

/* VMUL JitKernel */
template <typename T, platform::jit::cpu_isa_t isa, jit_block>
class VMulKernelImpl : public VMulKernel<T> {
 public:
  void Compute(const int n, const T* x, const T* y, T* z) override {
    for (int i = 0; i < n; ++i) {
      z[i] = x[i] * y[i];
    }
  }
};

#ifdef PADDLE_WITH_MKLML
#define VMUL_MKL_FLOAT(isa, block)                                             \
  template <>                                                                  \
  void VMulKernelImpl<float, isa, block>::Compute(const int n, const float* x, \
                                                  const float* y, float* z) {  \
    platform::dynload::vsMul(n, x, y, z);                                      \
  }

#define VMUL_MKL_DOUBLE(isa, block)                               \
  template <>                                                     \
  void VMulKernelImpl<double, isa, block>::Compute(               \
      const int n, const double* x, const double* y, double* z) { \
    platform::dynload::vdMul(n, x, y, z);                         \
  }

FOR_EACH_ISA(VMUL_MKL_FLOAT, kGT16);
FOR_EACH_ISA_BLOCK(VMUL_MKL_DOUBLE);
#endif

#define VMUL_INTRI8_FLOAT(isa)                                                \
  template <>                                                                 \
  void VMulKernelImpl<float, isa, kEQ8>::Compute(const int n, const float* x, \
                                                 const float* y, float* z) {  \
    __m256 tmpx, tmpy;                                                        \
    tmpx = _mm256_loadu_ps(x);                                                \
    tmpy = _mm256_loadu_ps(y);                                                \
    tmpx = _mm256_mul_ps(tmpx, tmpy);                                         \
    _mm256_storeu_ps(z, tmpx);                                                \
  }

// avx > for > mkl
#ifdef __AVX__
VMUL_INTRI8_FLOAT(jit::avx);
#endif
#ifdef __AVX2__
VMUL_INTRI8_FLOAT(jit::avx2);
#endif
#ifdef __AVX512F__
VMUL_INTRI8_FLOAT(jit::avx512f);
#endif

// TODO(TJ): eq16 test and complete avx512
#undef VMUL_INTRI8_FLOAT
#undef VMUL_MKL_FLOAT
#undef VMUL_MKL_DOUBLE

/* VADD JitKernel */
template <typename T, platform::jit::cpu_isa_t isa, jit_block>
class VAddKernelImpl : public VAddKernel<T> {
 public:
  void Compute(const int n, const T* x, const T* y, T* z) override {
    for (int i = 0; i < n; ++i) {
      z[i] = x[i] + y[i];
    }
  }
};

#ifdef PADDLE_WITH_MKLML
#define VADD_MKL_FLOAT(isa, block)                                             \
  template <>                                                                  \
  void VAddKernelImpl<float, isa, block>::Compute(const int n, const float* x, \
                                                  const float* y, float* z) {  \
    platform::dynload::vsAdd(n, x, y, z);                                      \
  }

#define VADD_MKL_DOUBLE(isa, block)                               \
  template <>                                                     \
  void VAddKernelImpl<double, isa, block>::Compute(               \
      const int n, const double* x, const double* y, double* z) { \
    platform::dynload::vdAdd(n, x, y, z);                         \
  }

FOR_EACH_ISA(VADD_MKL_FLOAT, kGT16);
FOR_EACH_ISA_BLOCK(VADD_MKL_DOUBLE);
#endif

#define VADD_INTRI8_FLOAT(isa)                                                \
  template <>                                                                 \
  void VAddKernelImpl<float, isa, kEQ8>::Compute(const int n, const float* x, \
                                                 const float* y, float* z) {  \
    __m256 tmpx, tmpy;                                                        \
    tmpx = _mm256_loadu_ps(x);                                                \
    tmpy = _mm256_loadu_ps(y);                                                \
    tmpx = _mm256_add_ps(tmpx, tmpy);                                         \
    _mm256_storeu_ps(z, tmpx);                                                \
  }
#ifdef __AVX__
VADD_INTRI8_FLOAT(jit::avx);
#endif
#ifdef __AVX2__
VADD_INTRI8_FLOAT(jit::avx2);
#endif
#ifdef __AVX512F__
VADD_INTRI8_FLOAT(jit::avx512f);
#endif
// TODO(TJ): eq16 test and complete avx512

#undef VADD_INTRI8_FLOAT
#undef VADD_MKL_FLOAT
#undef VADD_MKL_DOUBLE

/* VSCAL JitKernel */
template <typename T, platform::jit::cpu_isa_t isa, jit_block>
class VScalKernelImpl : public VScalKernel<T> {
 public:
  void Compute(const int n, const T a, const T* x, T* y) override {
    for (int i = 0; i < n; ++i) {
      y[i] = a * x[i];
    }
  }
  void Compute(const int n, const T a, T* x) override {
    for (int i = 0; i < n; ++i) {
      x[i] = a * x[i];
    }
  }
};

#ifdef PADDLE_WITH_MKLML
#define VSCAL_MKL_FLOAT(isa, block)                                            \
  template <>                                                                  \
  void VScalKernelImpl<float, isa, block>::Compute(const int n, const float a, \
                                                   float* x) {                 \
    platform::dynload::cblas_sscal(n, a, x, 1);                                \
  }

#define VSCAL_MKL_DOUBLE(isa, block)                 \
  template <>                                        \
  void VScalKernelImpl<double, isa, block>::Compute( \
      const int n, const double a, double* x) {      \
    platform::dynload::cblas_dscal(n, a, x, 1);      \
  }

FOR_EACH_ISA(VSCAL_MKL_FLOAT, kGT16);
FOR_EACH_ISA_BLOCK(VSCAL_MKL_DOUBLE);
#endif

#define VSCAL_INTRI8(isa)                                                     \
  template <>                                                                 \
  void VScalKernelImpl<float, isa, kEQ8>::Compute(const int n, const float a, \
                                                  const float* x, float* y) { \
    __m256 tmp;                                                               \
    __m256 scalar = _mm256_set1_ps(a);                                        \
    tmp = _mm256_loadu_ps(x);                                                 \
    tmp = _mm256_mul_ps(tmp, scalar);                                         \
    _mm256_storeu_ps(y, tmp);                                                 \
  }
#define VSCAL_INTRI8_INPLACE(isa)                                             \
  template <>                                                                 \
  void VScalKernelImpl<float, isa, kEQ8>::Compute(const int n, const float a, \
                                                  float* x) {                 \
    __m256 tmp;                                                               \
    __m256 scalar = _mm256_set1_ps(a);                                        \
    tmp = _mm256_loadu_ps(x);                                                 \
    tmp = _mm256_mul_ps(tmp, scalar);                                         \
    _mm256_storeu_ps(x, tmp);                                                 \
  }

#ifdef __AVX__
VSCAL_INTRI8(jit::avx);
VSCAL_INTRI8_INPLACE(jit::avx);
#endif
#ifdef __AVX2__
VSCAL_INTRI8(jit::avx2);
VSCAL_INTRI8_INPLACE(jit::avx2);
#endif
#ifdef __AVX512F__
VSCAL_INTRI8(jit::avx512f);
VSCAL_INTRI8_INPLACE(jit::avx512f);
#endif
// TODO(TJ): eq16 test and complete avx512

#undef VSCAL_INTRI8
#undef VSCAL_INTRI8_INPLACE
#undef VSCAL_MKL_FLOAT
#undef VSCAL_MKL_DOUBLE

REGISTER_BLAS_JITKERNEL(vmul, VMulKernel);
REGISTER_BLAS_JITKERNEL(vadd, VAddKernel);
REGISTER_BLAS_JITKERNEL(vscal, VScalKernel);

#undef FOR_EACH_ISA
#undef FOR_EACH_BLOCK
#undef FOR_EACH_ISA_BLOCK
#undef REGISTER_BLAS_JITKERNEL
#undef DEFINE_WITH_DTYPE
#undef SEARCH_ISA_BLOCK
#undef SEARCH_BLOCK
#undef NEW_IMPL

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
