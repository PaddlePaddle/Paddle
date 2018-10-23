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
  explicit VMulKernelImpl(int d) : VMulKernel<T>() { this->num_ = d; }
  void Compute(const T* x, const T* y, T* z) const override {
    for (int i = 0; i < this->num_; ++i) {
      z[i] = x[i] * y[i];
    }
  }
};

#ifdef PADDLE_WITH_MKLML
#define MKL_FLOAT(isa, block)                           \
  template <>                                           \
  void VMulKernelImpl<float, isa, block>::Compute(      \
      const float* x, const float* y, float* z) const { \
    platform::dynload::vsMul(this->num_, x, y, z);      \
  }

#define MKL_DOUBLE(isa, block)                             \
  template <>                                              \
  void VMulKernelImpl<double, isa, block>::Compute(        \
      const double* x, const double* y, double* z) const { \
    platform::dynload::vdMul(this->num_, x, y, z);         \
  }

FOR_EACH_ISA(MKL_FLOAT, kGT16);
FOR_EACH_ISA_BLOCK(MKL_DOUBLE);
#endif

#define INTRI8_FLOAT(isa)                               \
  template <>                                           \
  void VMulKernelImpl<float, isa, kEQ8>::Compute(       \
      const float* x, const float* y, float* z) const { \
    __m256 tmpx, tmpy;                                  \
    tmpx = _mm256_loadu_ps(x);                          \
    tmpy = _mm256_loadu_ps(y);                          \
    tmpx = _mm256_mul_ps(tmpx, tmpy);                   \
    _mm256_storeu_ps(z, tmpx);                          \
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
  explicit VAddKernelImpl(int d) : VAddKernel<T>() { this->num_ = d; }
  void Compute(const T* x, const T* y, T* z) const override {
    for (int i = 0; i < this->num_; ++i) {
      z[i] = x[i] + y[i];
    }
  }
};

#ifdef PADDLE_WITH_MKLML
#define MKL_FLOAT(isa, block)                           \
  template <>                                           \
  void VAddKernelImpl<float, isa, block>::Compute(      \
      const float* x, const float* y, float* z) const { \
    platform::dynload::vsAdd(this->num_, x, y, z);      \
  }

#define MKL_DOUBLE(isa, block)                             \
  template <>                                              \
  void VAddKernelImpl<double, isa, block>::Compute(        \
      const double* x, const double* y, double* z) const { \
    platform::dynload::vdAdd(this->num_, x, y, z);         \
  }

FOR_EACH_ISA(MKL_FLOAT, kGT16);
FOR_EACH_ISA_BLOCK(MKL_DOUBLE);
#endif

#define INTRI8_FLOAT(isa)                               \
  template <>                                           \
  void VAddKernelImpl<float, isa, kEQ8>::Compute(       \
      const float* x, const float* y, float* z) const { \
    __m256 tmpx, tmpy;                                  \
    tmpx = _mm256_loadu_ps(x);                          \
    tmpy = _mm256_loadu_ps(y);                          \
    tmpx = _mm256_add_ps(tmpx, tmpy);                   \
    _mm256_storeu_ps(z, tmpx);                          \
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
  explicit VScalKernelImpl(int d) : VScalKernel<T>() { this->num_ = d; }
  void Compute(const T a, const T* x, T* y) const override {
    for (int i = 0; i < this->num_; ++i) {
      y[i] = a * x[i];
    }
  }
  void Compute(const T a, T* x) const override {
    for (int i = 0; i < this->num_; ++i) {
      x[i] = a * x[i];
    }
  }
};

#ifdef PADDLE_WITH_MKLML
#define MKL_FLOAT(isa, block)                                               \
  template <>                                                               \
  void VScalKernelImpl<float, isa, block>::Compute(const float a, float* x) \
      const {                                                               \
    platform::dynload::cblas_sscal(this->num_, a, x, 1);                    \
  }

#define MKL_DOUBLE(isa, block)                                                 \
  template <>                                                                  \
  void VScalKernelImpl<double, isa, block>::Compute(const double a, double* x) \
      const {                                                                  \
    platform::dynload::cblas_dscal(this->num_, a, x, 1);                       \
  }

FOR_EACH_ISA(MKL_FLOAT, kGT16);
FOR_EACH_ISA_BLOCK(MKL_DOUBLE);
#endif

#define INTRI8_FLOAT(isa)                              \
  template <>                                          \
  void VScalKernelImpl<float, isa, kEQ8>::Compute(     \
      const float a, const float* x, float* y) const { \
    __m256 tmp;                                        \
    __m256 scalar = _mm256_set1_ps(a);                 \
    tmp = _mm256_loadu_ps(x);                          \
    tmp = _mm256_mul_ps(tmp, scalar);                  \
    _mm256_storeu_ps(y, tmp);                          \
  }
#define INTRI8_INPLACE_FLOAT(isa)                                          \
  template <>                                                              \
  void VScalKernelImpl<float, isa, kEQ8>::Compute(const float a, float* x) \
      const {                                                              \
    __m256 tmp;                                                            \
    __m256 scalar = _mm256_set1_ps(a);                                     \
    tmp = _mm256_loadu_ps(x);                                              \
    tmp = _mm256_mul_ps(tmp, scalar);                                      \
    _mm256_storeu_ps(x, tmp);                                              \
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

/* VAddBias JitKernel */
template <typename T, platform::jit::cpu_isa_t isa, jit_block>
class VAddBiasKernelImpl : public VAddBiasKernel<T> {
 public:
  explicit VAddBiasKernelImpl(int d) : VAddBiasKernel<T>() { this->num_ = d; }
  void Compute(const T a, const T* x, T* y) const override {
    for (int i = 0; i < this->num_; ++i) {
      y[i] = x[i] + a;
    }
  }
};

#define INTRI8_FLOAT(isa)                              \
  template <>                                          \
  void VAddBiasKernelImpl<float, isa, kEQ8>::Compute(  \
      const float a, const float* x, float* y) const { \
    __m256 tmp = _mm256_loadu_ps(x);                   \
    tmp = _mm256_add_ps(tmp, _mm256_set1_ps(a));       \
    _mm256_storeu_ps(y, tmp);                          \
  }

#define INTRI16_FLOAT(isa)                             \
  template <>                                          \
  void VAddBiasKernelImpl<float, isa, kEQ16>::Compute( \
      const float a, const float* x, float* y) const { \
    __m256 tmp0 = _mm256_loadu_ps(x);                  \
    __m256 tmp1 = _mm256_loadu_ps(x + 8);              \
    tmp0 = _mm256_add_ps(tmp0, _mm256_set1_ps(a));     \
    tmp1 = _mm256_add_ps(tmp1, _mm256_set1_ps(a));     \
    _mm256_storeu_ps(y, tmp0);                         \
    _mm256_storeu_ps(y + 8, tmp1);                     \
  }

#ifdef __AVX__
INTRI8_FLOAT(jit::avx);
INTRI16_FLOAT(jit::avx);
#endif
#ifdef __AVX2__
INTRI8_FLOAT(jit::avx2);
INTRI16_FLOAT(jit::avx2);
#endif
#ifdef __AVX512F__
INTRI8_FLOAT(jit::avx512f);
INTRI16_FLOAT(jit::avx512f);
#endif
// TODO(TJ): eq16 test and complete avx512

#undef INTRI8_FLOAT
#undef INTRI16_FLOAT

/* VRelu JitKernel */
template <typename T, platform::jit::cpu_isa_t isa, jit_block>
class VReluKernelImpl : public VReluKernel<T> {
 public:
  explicit VReluKernelImpl(int d) : VReluKernel<T>() { this->num_ = d; }
  void Compute(const T* x, T* y) const override {
    for (int i = 0; i < this->num_; ++i) {
      y[i] = x[i] > 0 ? x[i] : 0;
    }
  }
};

#define INTRI8_FLOAT(isa)                                                   \
  template <>                                                               \
  void VReluKernelImpl<float, isa, kEQ8>::Compute(const float* x, float* y) \
      const {                                                               \
    __m256 tmp = _mm256_loadu_ps(x);                                        \
    tmp = _mm256_max_ps(tmp, _mm256_setzero_ps());                          \
    _mm256_storeu_ps(y, tmp);                                               \
  }

#define INTRI16_FLOAT(isa)                                                   \
  template <>                                                                \
  void VReluKernelImpl<float, isa, kEQ16>::Compute(const float* x, float* y) \
      const {                                                                \
    __m256 zeros = _mm256_setzero_ps();                                      \
    __m256 tmp0 = _mm256_loadu_ps(x);                                        \
    __m256 tmp1 = _mm256_loadu_ps(x + 8);                                    \
    tmp0 = _mm256_max_ps(tmp0, zeros);                                       \
    tmp1 = _mm256_max_ps(tmp1, zeros);                                       \
    _mm256_storeu_ps(y, tmp0);                                               \
    _mm256_storeu_ps(y + 8, tmp1);                                           \
  }

#define INTRI_GT8LT16_FLOAT(isa)                                        \
  template <>                                                           \
  VReluKernelImpl<float, isa, kGT8LT16>::VReluKernelImpl(int d)         \
      : VReluKernel<float>() {                                          \
    this->num_ = d;                                                     \
    this->end_ = AVX_FLOAT_BLOCK;                                       \
    this->rest_ = d - AVX_FLOAT_BLOCK;                                  \
  }                                                                     \
  template <>                                                           \
  void VReluKernelImpl<float, isa, kGT8LT16>::Compute(const float* x,   \
                                                      float* y) const { \
    __m256 zeros = _mm256_setzero_ps();                                 \
    __m256 tmp0 = _mm256_loadu_ps(x);                                   \
    __m256 tmp1 = _mm256_loadu_ps(x + this->rest_);                     \
    tmp0 = _mm256_max_ps(tmp0, zeros);                                  \
    tmp1 = _mm256_max_ps(tmp1, zeros);                                  \
    _mm256_storeu_ps(y, tmp0);                                          \
    _mm256_storeu_ps(y + this->rest_, tmp1);                            \
  }

#define INTRI_GT16_FLOAT(isa)                                                \
  template <>                                                                \
  VReluKernelImpl<float, isa, kGT16>::VReluKernelImpl(int d)                 \
      : VReluKernel<float>() {                                               \
    this->num_ = d;                                                          \
    this->end_ = d - d % AVX_FLOAT_BLOCK;                                    \
    this->rest_ = d - AVX_FLOAT_BLOCK;                                       \
  }                                                                          \
  template <>                                                                \
  void VReluKernelImpl<float, isa, kGT16>::Compute(const float* x, float* y) \
      const {                                                                \
    __m256 zeros = _mm256_setzero_ps();                                      \
    for (int i = 0; i < this->end_; i += AVX_FLOAT_BLOCK) {                  \
      __m256 tmp = _mm256_loadu_ps(x + i);                                   \
      tmp = _mm256_max_ps(tmp, zeros);                                       \
      _mm256_storeu_ps(y + i, tmp);                                          \
    }                                                                        \
    __m256 tmp = _mm256_loadu_ps(x + this->rest_);                           \
    tmp = _mm256_max_ps(tmp, zeros);                                         \
    _mm256_storeu_ps(y + this->rest_, tmp);                                  \
  }

#ifdef __AVX__
INTRI8_FLOAT(jit::avx);
INTRI16_FLOAT(jit::avx);
INTRI_GT8LT16_FLOAT(jit::avx);
INTRI_GT16_FLOAT(jit::avx);
#endif
#ifdef __AVX2__
INTRI8_FLOAT(jit::avx2);
INTRI16_FLOAT(jit::avx2);
INTRI_GT8LT16_FLOAT(jit::avx2);
INTRI_GT16_FLOAT(jit::avx2);
#endif
#ifdef __AVX512F__
// TODO(TJ): refine avx512
INTRI8_FLOAT(jit::avx512f);
INTRI16_FLOAT(jit::avx512f);
INTRI_GT8LT16_FLOAT(jit::avx512f);
INTRI_GT16_FLOAT(jit::avx512f);
#endif

#undef INTRI8_FLOAT
#undef INTRI16_FLOAT
#undef INTRI_GT8LT16_FLOAT
#undef INTRI_GT16_FLOAT

/* An empty JitKernel */
template <typename T, platform::jit::cpu_isa_t isa, jit_block>
class VIdentityKernelImpl : public VIdentityKernel<T> {
 public:
  explicit VIdentityKernelImpl(int d) : VIdentityKernel<T>() { this->num_ = d; }
  void Compute(const T* x, T* y) const override {}
};

/* VAddRelu JitKernel */
template <typename T, platform::jit::cpu_isa_t isa, jit_block>
class VAddReluKernelImpl : public VAddReluKernel<T> {
 public:
  explicit VAddReluKernelImpl(int d) : VAddReluKernel<T>() { this->num_ = d; }
  void Compute(const T* x, const T* y, T* z) const override {
    for (int i = 0; i < this->num_; ++i) {
      z[i] = x[i] + y[i];
      z[i] = z[i] > 0 ? z[i] : 0;
    }
  }
};

#define INTRI8_FLOAT(isa)                               \
  template <>                                           \
  void VAddReluKernelImpl<float, isa, kEQ8>::Compute(   \
      const float* x, const float* y, float* z) const { \
    __m256 tmpx = _mm256_loadu_ps(x);                   \
    __m256 tmpy = _mm256_loadu_ps(y);                   \
    tmpy = _mm256_add_ps(tmpx, tmpy);                   \
    tmpy = _mm256_max_ps(tmpy, _mm256_setzero_ps());    \
    _mm256_storeu_ps(z, tmpy);                          \
  }

#define INTRI16_FLOAT(isa)                              \
  template <>                                           \
  void VAddReluKernelImpl<float, isa, kEQ16>::Compute(  \
      const float* x, const float* y, float* z) const { \
    __m256 zeros = _mm256_setzero_ps();                 \
    __m256 tmp0 = _mm256_loadu_ps(x);                   \
    __m256 tmp1 = _mm256_loadu_ps(y);                   \
    tmp0 = _mm256_add_ps(tmp0, tmp1);                   \
    tmp0 = _mm256_max_ps(tmp0, zeros);                  \
    tmp1 = _mm256_loadu_ps(x + 8);                      \
    __m256 tmp2 = _mm256_loadu_ps(y + 8);               \
    tmp1 = _mm256_add_ps(tmp1, tmp2);                   \
    tmp1 = _mm256_max_ps(tmp1, zeros);                  \
    _mm256_storeu_ps(z, tmp0);                          \
    _mm256_storeu_ps(z + 8, tmp1);                      \
  }

#define INTRI_COMMON_FLOAT(isa, block)                             \
  template <>                                                      \
  VAddReluKernelImpl<float, isa, block>::VAddReluKernelImpl(int d) \
      : VAddReluKernel<float>() {                                  \
    this->num_ = d;                                                \
    this->end_ = d - d % AVX_FLOAT_BLOCK;                          \
    this->rest_ = d - this->end_;                                  \
  }                                                                \
  template <>                                                      \
  void VAddReluKernelImpl<float, isa, block>::Compute(             \
      const float* x, const float* y, float* z) const {            \
    __m256 zeros = _mm256_setzero_ps();                            \
    for (int i = 0; i < this->end_; i += AVX_FLOAT_BLOCK) {        \
      __m256 tmpx = _mm256_loadu_ps(x + i);                        \
      __m256 tmpy = _mm256_loadu_ps(y + i);                        \
      tmpy = _mm256_add_ps(tmpx, tmpy);                            \
      tmpy = _mm256_max_ps(tmpy, zeros);                           \
      _mm256_storeu_ps(z + i, tmpy);                               \
    }                                                              \
    for (int i = this->end_; i < this->num_; ++i) {                \
      z[i] = x[i] + y[i];                                          \
      z[i] = z[i] > 0 ? z[i] : 0;                                  \
    }                                                              \
  }

#ifdef __AVX__
INTRI8_FLOAT(jit::avx);
INTRI16_FLOAT(jit::avx);
INTRI_COMMON_FLOAT(jit::avx, kGT16);
#endif
#ifdef __AVX2__
INTRI8_FLOAT(jit::avx2);
INTRI16_FLOAT(jit::avx2);
INTRI_COMMON_FLOAT(jit::avx2, kGT16);
#endif
#ifdef __AVX512F__
// TODO(TJ): refine avx512
INTRI8_FLOAT(jit::avx512f);
INTRI16_FLOAT(jit::avx512f);
INTRI_COMMON_FLOAT(jit::avx512f, kGT16);
#endif

#undef INTRI8_FLOAT
#undef INTRI16_FLOAT
#undef INTRI_COMMON_FLOAT

REGISTER_JITKERNEL(vmul, VMulKernel);
REGISTER_JITKERNEL(vadd, VAddKernel);
REGISTER_JITKERNEL(vscal, VScalKernel);
REGISTER_JITKERNEL(vaddb, VAddBiasKernel);
REGISTER_JITKERNEL(vrelu, VReluKernel);
REGISTER_JITKERNEL(vaddrelu, VAddReluKernel);
REGISTER_JITKERNEL(videntity, VIdentityKernel);

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
