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

#define JITKERNEL_NEW_ACT_IMPL(ker, dtype, isa, k) \
  p = std::dynamic_pointer_cast<ker<dtype>>(       \
      std::make_shared<ker##Impl<dtype, isa, k>>(d))

REGISTER_JITKERNEL_ARGS(vsigmoid, VSigmoidKernel, JITKERNEL_DECLARE,
                        JITKERNEL_KEY, JITKERNEL_NEW_ACT_IMPL);

#undef JITKERNEL_NEW_ACT_IMPL
}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
