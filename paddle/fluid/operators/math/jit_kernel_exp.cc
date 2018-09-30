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

#ifdef __AVX__
namespace detail {
__m256 Exp(__m256 a);
}  // namespace detail
#endif

namespace jitkernel {
namespace jit = platform::jit;

/* VExp JitKernel */
template <typename T, jit::cpu_isa_t isa, jit_block>
class VExpKernelImpl : public VExpKernel<T> {
 public:
  void Compute(const int n, const T* x, T* y) const override {
    for (int i = 0; i < n; ++i) {
      y[i] = std::exp(x[i]);
    }
  }
};

#ifdef PADDLE_WITH_MKLML
#define MKL_FLOAT(isa, block)                                                  \
  template <>                                                                  \
  void VExpKernelImpl<float, isa, block>::Compute(const int n, const float* x, \
                                                  float* y) const {            \
    platform::dynload::vsExp(n, x, y);                                         \
  }

#define MKL_DOUBLE(isa, block)                         \
  template <>                                          \
  void VExpKernelImpl<double, isa, block>::Compute(    \
      const int n, const double* x, double* y) const { \
    platform::dynload::vdExp(n, x, y);                 \
  }
FOR_EACH_ISA(MKL_FLOAT, kLT8);
FOR_EACH_ISA(MKL_FLOAT, kGT8LT16);
FOR_EACH_ISA(MKL_FLOAT, kGT16);
FOR_EACH_ISA_BLOCK(MKL_DOUBLE);
#endif

#define INTRI8_FLOAT(isa)                                                     \
  template <>                                                                 \
  void VExpKernelImpl<float, isa, kEQ8>::Compute(const int n, const float* x, \
                                                 float* y) const {            \
    __m256 tmp = _mm256_loadu_ps(x);                                          \
    _mm256_storeu_ps(y, detail::Exp(tmp));                                    \
  }

#define INTRI16_FLOAT(isa)                                                     \
  template <>                                                                  \
  void VExpKernelImpl<float, isa, kEQ16>::Compute(const int n, const float* x, \
                                                  float* y) const {            \
    __m256 tmp0 = _mm256_loadu_ps(x);                                          \
    __m256 tmp1 = _mm256_loadu_ps(x + 8);                                      \
    tmp0 = detail::Exp(tmp0);                                                  \
    tmp1 = detail::Exp(tmp1);                                                  \
    _mm256_storeu_ps(y, tmp0);                                                 \
    _mm256_storeu_ps(y + 8, tmp1);                                             \
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
#undef MKL_FLOAT
#undef MKL_DOUBLE

REGISTER_JITKERNEL(vexp, VExpKernel);

/* VSigmoid JitKernel */
template <typename T, jit::cpu_isa_t isa, jit_block>
class VSigmoidKernelImpl : public VSigmoidKernel<T> {
 public:
  explicit VSigmoidKernelImpl(int d) : VSigmoidKernel<T>() {
    vexp_ = KernelPool::Instance().template Get<VExpKernel<T>>(d);
  }
  void Compute(const int n, const T* x, T* y) const override {
    const T min = SIGMOID_THRESHOLD_MIN;
    const T max = SIGMOID_THRESHOLD_MAX;
    for (int i = 0; i < n; ++i) {
      y[i] = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
      y[i] = static_cast<T>(0) - y[i];
    }
    vexp_->Compute(n, y, y);
    for (int i = 0; i < n; ++i) {
      y[i] = static_cast<T>(1) / (static_cast<T>(1) + y[i]);
    }
  }

 private:
  std::shared_ptr<const VExpKernel<T>> vexp_;
};

#define INTRI_SIGMOID(tmp, min, max)              \
  tmp = _mm256_max_ps(tmp, min);                  \
  tmp = _mm256_min_ps(tmp, max);                  \
  tmp = _mm256_sub_ps(_mm256_set1_ps(0.0f), tmp); \
  tmp = detail::Exp(tmp);                         \
  tmp = _mm256_add_ps(_mm256_set1_ps(1.0f), tmp); \
  tmp = _mm256_div_ps(_mm256_set1_ps(1.0f), tmp)

#define INTRI8_FLOAT(isa)                               \
  template <>                                           \
  void VSigmoidKernelImpl<float, isa, kEQ8>::Compute(   \
      const int n, const float* x, float* y) const {    \
    __m256 max = _mm256_set1_ps(SIGMOID_THRESHOLD_MAX); \
    __m256 min = _mm256_set1_ps(SIGMOID_THRESHOLD_MIN); \
    __m256 tmp = _mm256_loadu_ps(x);                    \
    INTRI_SIGMOID(tmp, min, max);                       \
    _mm256_storeu_ps(y, tmp);                           \
  }

#define INTRI16_FLOAT(isa)                              \
  template <>                                           \
  void VSigmoidKernelImpl<float, isa, kEQ16>::Compute(  \
      const int n, const float* x, float* y) const {    \
    __m256 max = _mm256_set1_ps(SIGMOID_THRESHOLD_MAX); \
    __m256 min = _mm256_set1_ps(SIGMOID_THRESHOLD_MIN); \
    __m256 tmp0 = _mm256_loadu_ps(x);                   \
    __m256 tmp1 = _mm256_loadu_ps(x + 8);               \
    INTRI_SIGMOID(tmp0, min, max);                      \
    INTRI_SIGMOID(tmp1, min, max);                      \
    _mm256_storeu_ps(y, tmp0);                          \
    _mm256_storeu_ps(y + 8, tmp1);                      \
  }

#define INTRI_GT8LT16_FLOAT(isa)                                   \
  template <>                                                      \
  void VSigmoidKernelImpl<float, isa, kGT8LT16>::Compute(          \
      const int n, const float* x, float* y) const {               \
    __m256 max = _mm256_set1_ps(SIGMOID_THRESHOLD_MAX);            \
    __m256 min = _mm256_set1_ps(SIGMOID_THRESHOLD_MIN);            \
    __m256 tmp = _mm256_loadu_ps(x);                               \
    INTRI_SIGMOID(tmp, min, max);                                  \
    _mm256_storeu_ps(y, tmp);                                      \
    const float min_ = SIGMOID_THRESHOLD_MIN;                      \
    const float max_ = SIGMOID_THRESHOLD_MAX;                      \
    for (int i = AVX_FLOAT_BLOCK; i < n; ++i) {                    \
      y[i] = (x[i] < min_) ? min_ : ((x[i] > max_) ? max_ : x[i]); \
      y[i] = 0.f - y[i];                                           \
    }                                                              \
    vexp_->Compute(n - AVX_FLOAT_BLOCK, y + AVX_FLOAT_BLOCK,       \
                   y + AVX_FLOAT_BLOCK);                           \
    for (int i = AVX_FLOAT_BLOCK; i < n; ++i) {                    \
      y[i] = 1.f / (1.f + y[i]);                                   \
    }                                                              \
  }

#define INTRI_GT16_FLOAT(isa)                                      \
  template <>                                                      \
  void VSigmoidKernelImpl<float, isa, kGT16>::Compute(             \
      const int n, const float* x, float* y) const {               \
    __m256 max = _mm256_set1_ps(SIGMOID_THRESHOLD_MAX);            \
    __m256 min = _mm256_set1_ps(SIGMOID_THRESHOLD_MIN);            \
    const int rest = n % AVX_FLOAT_BLOCK;                          \
    const int end = n - rest;                                      \
    for (int i = 0; i < end; i += AVX_FLOAT_BLOCK) {               \
      __m256 tmp = _mm256_loadu_ps(x + i);                         \
      INTRI_SIGMOID(tmp, min, max);                                \
      _mm256_storeu_ps(y + i, tmp);                                \
    }                                                              \
    const float min_ = SIGMOID_THRESHOLD_MIN;                      \
    const float max_ = SIGMOID_THRESHOLD_MAX;                      \
    for (int i = end; i < n; ++i) {                                \
      y[i] = (x[i] < min_) ? min_ : ((x[i] > max_) ? max_ : x[i]); \
      y[i] = 0.f - y[i];                                           \
    }                                                              \
    vexp_->Compute(rest, y + end, y + end);                        \
    for (int i = end; i < n; ++i) {                                \
      y[i] = 1.f / (1.f + y[i]);                                   \
    }                                                              \
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
// INTRI_GT8LT16_FLOAT(jit::avx2);
// INTRI_GT16_FLOAT(jit::avx2);
#endif
#ifdef __AVX512F__
INTRI8_FLOAT(jit::avx512f);
INTRI16_FLOAT(jit::avx512f);
// INTRI_GT8LT16_FLOAT(jit::avx512f);
// INTRI_GT16_FLOAT(jit::avx512f);
#endif

#undef INTRI8_FLOAT
#undef INTRI16_FLOAT
#undef INTRI_GT8LT16_FLOAT
#undef INTRI_GT16_FLOAT
#undef INTRI_VSIGMOID

#define JITKERNEL_NEW_ACT_IMPL(ker, dtype, isa, k) \
  p = std::dynamic_pointer_cast<ker<dtype>>(       \
      std::make_shared<ker##Impl<dtype, isa, k>>(d))

REGISTER_JITKERNEL_ARGS(vsigmoid, VSigmoidKernel, JITKERNEL_DECLARE,
                        JITKERNEL_KEY, JITKERNEL_NEW_ACT_IMPL);

/* VTanh JitKernel */
template <typename T, jit::cpu_isa_t isa, jit_block>
class VTanhKernelImpl : public VTanhKernel<T> {
 public:
  explicit VTanhKernelImpl(int d) : VTanhKernel<T>() {
    vscal_ = KernelPool::Instance().template Get<VScalKernel<T>>(d);
    vsigmoid_ = KernelPool::Instance().template Get<VSigmoidKernel<T>>(d);
    vaddbias_ = KernelPool::Instance().template Get<VAddBiasKernel<T>>(d);
  }
  void Compute(const int n, const T* x, T* y) const override {
    vscal_->Compute(n, static_cast<T>(2), x, y);
    vsigmoid_->Compute(n, y, y);
    vscal_->Compute(n, static_cast<T>(2), y);
    vaddbias_->Compute(n, static_cast<T>(-1), y, y);
  }

 private:
  std::shared_ptr<const VScalKernel<T>> vscal_;
  std::shared_ptr<const VSigmoidKernel<T>> vsigmoid_;
  std::shared_ptr<const VAddBiasKernel<T>> vaddbias_;
};

#define INTRI_VTANH(tmp)                                   \
  tmp = _mm256_mul_ps(_mm256_set1_ps(-2.0f), tmp);         \
  tmp = _mm256_min_ps(tmp, _mm256_set1_ps(EXP_MAX_INPUT)); \
  tmp = detail::Exp(tmp);                                  \
  tmp = _mm256_add_ps(_mm256_set1_ps(1.0f), tmp);          \
  tmp = _mm256_div_ps(_mm256_set1_ps(2.0f), tmp);          \
  tmp = _mm256_sub_ps(tmp, _mm256_set1_ps(1.0f))

#define INTRI8_FLOAT(isa)                                                      \
  template <>                                                                  \
  void VTanhKernelImpl<float, isa, kEQ8>::Compute(const int n, const float* x, \
                                                  float* y) const {            \
    __m256 tmp = _mm256_loadu_ps(x);                                           \
    INTRI_VTANH(tmp);                                                          \
    _mm256_storeu_ps(y, tmp);                                                  \
  }

#define INTRI16_FLOAT(isa)                           \
  template <>                                        \
  void VTanhKernelImpl<float, isa, kEQ16>::Compute(  \
      const int n, const float* x, float* y) const { \
    __m256 tmp0 = _mm256_loadu_ps(x);                \
    __m256 tmp1 = _mm256_loadu_ps(x + 8);            \
    INTRI_VTANH(tmp0);                               \
    INTRI_VTANH(tmp1);                               \
    _mm256_storeu_ps(y, tmp0);                       \
    _mm256_storeu_ps(y + 8, tmp1);                   \
  }

#define INTRI_GT8LT16_FLOAT(isa)                       \
  template <>                                          \
  void VTanhKernelImpl<float, isa, kGT8LT16>::Compute( \
      const int n, const float* x, float* y) const {   \
    __m256 tmp = _mm256_loadu_ps(x);                   \
    INTRI_VTANH(tmp);                                  \
    _mm256_storeu_ps(y, tmp);                          \
    x += AVX_FLOAT_BLOCK;                              \
    y += AVX_FLOAT_BLOCK;                              \
    const int rest = n - AVX_FLOAT_BLOCK;              \
    vscal_->Compute(rest, 2.f, x, y);                  \
    vsigmoid_->Compute(rest, y, y);                    \
    vscal_->Compute(rest, 2.f, y);                     \
    vaddbias_->Compute(rest, -1.f, y, y);              \
  }

#define INTRI_GT16_FLOAT(isa)                        \
  template <>                                        \
  void VTanhKernelImpl<float, isa, kGT16>::Compute(  \
      const int n, const float* x, float* y) const { \
    const int rest = n % AVX_FLOAT_BLOCK;            \
    const int end = n - rest;                        \
    for (int i = 0; i < end; i += AVX_FLOAT_BLOCK) { \
      __m256 tmp = _mm256_loadu_ps(x + i);           \
      INTRI_VTANH(tmp);                              \
      _mm256_storeu_ps(y + i, tmp);                  \
    }                                                \
    x += end;                                        \
    y += end;                                        \
    vscal_->Compute(rest, 2.f, x, y);                \
    vsigmoid_->Compute(rest, y, y);                  \
    vscal_->Compute(rest, 2.f, y);                   \
    vaddbias_->Compute(rest, -1.f, y, y);            \
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
// maybe use avx at gt8lt16 and gt16
#endif
#ifdef __AVX512F__
INTRI8_FLOAT(jit::avx512f);
INTRI16_FLOAT(jit::avx512f);
// maybe use avx at gt8lt16 and gt16
#endif

#undef INTRI8_FLOAT
#undef INTRI16_FLOAT
#undef INTRI_GT8LT16_FLOAT
#undef INTRI_GT16_FLOAT
#undef INTRI_VTANH

REGISTER_JITKERNEL_ARGS(vtanh, VTanhKernel, JITKERNEL_DECLARE, JITKERNEL_KEY,
                        JITKERNEL_NEW_ACT_IMPL);

#undef JITKERNEL_NEW_ACT_IMPL

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
