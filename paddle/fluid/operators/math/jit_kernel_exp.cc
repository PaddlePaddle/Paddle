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
#include <cmath>  // for exp
#include <string>
#include "paddle/fluid/operators/math/jit_kernel_macro.h"

#ifdef PADDLE_WITH_XBYAK
#include "paddle/fluid/operators/math/jit_code.h"
#endif

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

// TODO(TJ): move refer codes to one file
// Refer code only focus on correctness
template <typename T>
void VExpRefer(const T* x, T* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = std::exp(x[i]);
  }
}

template <typename T>
void VSigmoidRefer(const T* x, T* y, int n) {
  // y = 1 / (1 + e^-x)
  const T min = SIGMOID_THRESHOLD_MIN;
  const T max = SIGMOID_THRESHOLD_MAX;
  for (int i = 0; i < n; ++i) {
    T tmp = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
    y[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-tmp));
  }
}

template <typename T>
void VTanhRefer(const T* x, T* y, int n) {
  // y = 2 * sigmoid(2x) - 1
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(2) * x[i];
  }
  VSigmoidRefer(y, y, n);
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(2) * y[i] - static_cast<T>(1);
  }
}

#ifdef PADDLE_WITH_MKLML
// try to use MKL to speedup
template <typename T>
void VExpMKL(const T* x, T* y, int n);

template <>
void VExpMKL<float>(const float* x, float* y, int n) {
  platform::dynload::vsExp(n, x, y);
}

template <>
void VExpMKL<double>(const double* x, double* y, int n) {
  platform::dynload::vdExp(n, x, y);
}

template <typename T>
void VSigmoidMKL(const T* x, T* y, int n) {
  const T min = SIGMOID_THRESHOLD_MIN;
  const T max = SIGMOID_THRESHOLD_MAX;
  for (int i = 0; i < n; ++i) {
    y[i] = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
    y[i] = static_cast<T>(0) - y[i];
  }
  VExpMKL(y, y, n);
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(1) / (static_cast<T>(1) + y[i]);
  }
}

template <typename T>
void VTanhMKL(const T* x, T* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(2) * x[i];
  }
  VSigmoidMKL(y, y, n);
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(2) * y[i] - static_cast<T>(1);
  }
}
#endif

/* VExp JitKernel */
template <typename T>
class VExpKernelImpl : public VExpKernel<T> {
 public:
  JITKERNEL_DECLARE_STATIC_FUNC;
  explicit VExpKernelImpl(int d) : VExpKernel<T>() {
#ifdef PADDLE_WITH_XBYAK
    if (useJIT(d)) {
      size_t sz = 96 + d / YMM_FLOAT_BLOCK * 70 * 8;
      jitcode_.reset(new gen::VActJitCode(d, gen::operand_type::exp,
                                          sz > 4096 ? sz : 4096));
      this->Compute = jitcode_->getCode<void (*)(const T*, T*, int)>();
      return;
    }
#endif
#ifdef PADDLE_WITH_MKLML
    if (useMKL(d)) {
      this->Compute = VExpMKL<T>;
      return;
    }
#endif
    this->Compute = VExpRefer<T>;
  }

#ifdef PADDLE_WITH_XBYAK

 private:
  std::unique_ptr<gen::VActJitCode> jitcode_{nullptr};
#endif
};

#ifdef PADDLE_WITH_XBYAK
template <>
bool VExpKernelImpl<float>::useJIT(int d) {
  return gen::VActJitCode::init(d, gen::operand_type::exp);
}
#endif

#ifdef PADDLE_WITH_MKLML
template <>
bool VExpKernelImpl<float>::useMKL(int d) {
  return d > 512;
}

template <>
bool VExpKernelImpl<double>::useMKL(int d) {
  return true;
}

#endif

/* VSigmoid JitKernel */
template <typename T>
class VSigmoidKernelImpl : public VSigmoidKernel<T> {
 public:
  JITKERNEL_DECLARE_STATIC_FUNC;
  explicit VSigmoidKernelImpl(int d) : VSigmoidKernel<T>() {
#ifdef PADDLE_WITH_XBYAK
    if (useJIT(d)) {
      size_t sz = 96 + d / YMM_FLOAT_BLOCK * 82 * 8;
      jitcode_.reset(new gen::VActJitCode(d, gen::operand_type::sigmoid,
                                          sz > 4096 ? sz : 4096));
      this->Compute = jitcode_->getCode<void (*)(const T*, T*, int)>();
      return;
    }
#endif

#ifdef PADDLE_WITH_MKLML
    // strictly it's a better impl with MKL, then is refer
    if (useMKL(d)) {
      this->Compute = VSigmoidMKL<T>;
      return;
    }
#endif
    this->Compute = VSigmoidRefer<T>;
  }

#ifdef PADDLE_WITH_XBYAK

 private:
  std::unique_ptr<gen::VActJitCode> jitcode_{nullptr};
#endif
};

#ifdef PADDLE_WITH_XBYAK
template <>
bool VSigmoidKernelImpl<float>::useJIT(int d) {
  return gen::VActJitCode::init(d, gen::operand_type::sigmoid);
}
#endif

#ifdef PADDLE_WITH_MKLML
template <>
bool VSigmoidKernelImpl<float>::useMKL(int d) {
  return d > 512;
}

template <>
bool VSigmoidKernelImpl<double>::useMKL(int d) {
  return true;
}
#endif

/* VTanh JitKernel */
template <typename T>
class VTanhKernelImpl : public VTanhKernel<T> {
 public:
  JITKERNEL_DECLARE_STATIC_FUNC;
  explicit VTanhKernelImpl(int d) : VTanhKernel<T>() {
#ifdef PADDLE_WITH_XBYAK
    if (useJIT(d)) {
      size_t sz = 96 + d / YMM_FLOAT_BLOCK * 84 * 8;
      jitcode_.reset(new gen::VActJitCode(d, gen::operand_type::tanh,
                                          sz > 4096 ? sz : 4096));
      this->Compute = jitcode_->getCode<void (*)(const T*, T*, int)>();
      return;
    }
#endif

#ifdef PADDLE_WITH_MKLML
    // strictly it's a better impl with MKL, then is refer
    if (useMKL(d)) {
      this->Compute = VTanhMKL<T>;
      return;
    }
#endif
    this->Compute = VTanhRefer<T>;
  }

#ifdef PADDLE_WITH_XBYAK

 private:
  std::unique_ptr<gen::VActJitCode> jitcode_{nullptr};
#endif
};

#ifdef PADDLE_WITH_XBYAK
template <>
bool VTanhKernelImpl<float>::useJIT(int d) {
  return gen::VActJitCode::init(d, gen::operand_type::tanh);
}
#endif

#ifdef PADDLE_WITH_MKLML
template <>
bool VTanhKernelImpl<float>::useMKL(int d) {
  return d > 512;
}

template <>
bool VTanhKernelImpl<double>::useMKL(int d) {
  return true;
}
#endif

REGISTER_JITKERNEL(vexp, VExpKernel);
REGISTER_JITKERNEL(vsigmoid, VSigmoidKernel);
REGISTER_JITKERNEL(vtanh, VTanhKernel);

namespace detail {

#ifdef __AVX__

#define ALIGN32 __attribute__((aligned(32)))

#define _PS256_CONST(Name, Val)                                      \
  static const float _ps256_##Name[8] ALIGN32 = {Val, Val, Val, Val, \
                                                 Val, Val, Val, Val}

#define _PI256_CONST(Name, Val)                                    \
  static const int _pi256_##Name[8] ALIGN32 = {Val, Val, Val, Val, \
                                               Val, Val, Val, Val}

_PI256_CONST(0x7f, 0x7f);
_PS256_CONST(one, 1.f);
_PS256_CONST(0p5, 0.5f);
_PS256_CONST(exp_hi, 88.3762626647949f);
_PS256_CONST(exp_lo, -88.3762626647949f);
_PS256_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS256_CONST(cephes_exp_C1, 0.693359375);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4);
_PS256_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1);

typedef union imm_xmm_union {
  __m256i imm;
  __m128i xmm[2];
} imm_xmm_union;

#define COPY_IMM_TO_XMM(imm_, xmm0_, xmm1_) \
  {                                         \
    imm_xmm_union u ALIGN32;                \
    u.imm = imm_;                           \
    xmm0_ = u.xmm[0];                       \
    xmm1_ = u.xmm[1];                       \
  }

#define COPY_XMM_TO_IMM(xmm0_, xmm1_, imm_) \
  {                                         \
    imm_xmm_union u ALIGN32;                \
    u.xmm[0] = xmm0_;                       \
    u.xmm[1] = xmm1_;                       \
    imm_ = u.imm;                           \
  }

#define AVX2_BITOP_USING_SSE2(fn)                           \
  static inline __m256i avx2_mm256_##fn(__m256i x, int y) { \
    /* use SSE2 to perform the bitop AVX2 */                \
    __m128i x1, x2;                                         \
    __m256i ret;                                            \
    COPY_IMM_TO_XMM(x, x1, x2);                             \
    x1 = _mm_##fn(x1, y);                                   \
    x2 = _mm_##fn(x2, y);                                   \
    COPY_XMM_TO_IMM(x1, x2, ret);                           \
    return ret;                                             \
  }

#define AVX2_INTOP_USING_SSE2(fn)                                    \
  static inline __m256i avx2_mm256_add_epi32(__m256i x, __m256i y) { \
    /* use SSE2 to perform the AVX2 integer operation */             \
    __m128i x1, x2;                                                  \
    __m128i y1, y2;                                                  \
    __m256i ret;                                                     \
    COPY_IMM_TO_XMM(x, x1, x2);                                      \
    COPY_IMM_TO_XMM(y, y1, y2);                                      \
    x1 = _mm_##fn(x1, y1);                                           \
    x2 = _mm_##fn(x2, y2);                                           \
    COPY_XMM_TO_IMM(x1, x2, ret);                                    \
    return ret;                                                      \
  }

AVX2_BITOP_USING_SSE2(slli_epi32);
AVX2_INTOP_USING_SSE2(add_epi32);

#define AVXEXP_BASE                                                            \
  __m256 tmp = _mm256_setzero_ps(), fx;                                        \
  __m256 one = *reinterpret_cast<const __m256*>(_ps256_one);                   \
  __m256i imm0;                                                                \
  x = _mm256_min_ps(x, *reinterpret_cast<const __m256*>(_ps256_exp_hi));       \
  x = _mm256_max_ps(x, *reinterpret_cast<const __m256*>(_ps256_exp_lo));       \
  /* express exp(x) as exp(g + n*log(2)) */                                    \
  fx = _mm256_mul_ps(x,                                                        \
                     *reinterpret_cast<const __m256*>(_ps256_cephes_LOG2EF));  \
  fx = _mm256_add_ps(fx, *reinterpret_cast<const __m256*>(_ps256_0p5));        \
  tmp = _mm256_floor_ps(fx);                                                   \
  /* if greater, substract 1 */                                                \
  __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);                            \
  mask = _mm256_and_ps(mask, one);                                             \
  fx = _mm256_sub_ps(tmp, mask);                                               \
  tmp = _mm256_mul_ps(fx,                                                      \
                      *reinterpret_cast<const __m256*>(_ps256_cephes_exp_C1)); \
  __m256 z = _mm256_mul_ps(                                                    \
      fx, *reinterpret_cast<const __m256*>(_ps256_cephes_exp_C2));             \
  x = _mm256_sub_ps(x, tmp);                                                   \
  x = _mm256_sub_ps(x, z);                                                     \
  z = _mm256_mul_ps(x, x);                                                     \
  __m256 y = *reinterpret_cast<const __m256*>(_ps256_cephes_exp_p0);           \
  y = _mm256_mul_ps(y, x);                                                     \
  y = _mm256_add_ps(y,                                                         \
                    *reinterpret_cast<const __m256*>(_ps256_cephes_exp_p1));   \
  y = _mm256_mul_ps(y, x);                                                     \
  y = _mm256_add_ps(y,                                                         \
                    *reinterpret_cast<const __m256*>(_ps256_cephes_exp_p2));   \
  y = _mm256_mul_ps(y, x);                                                     \
  y = _mm256_add_ps(y,                                                         \
                    *reinterpret_cast<const __m256*>(_ps256_cephes_exp_p3));   \
  y = _mm256_mul_ps(y, x);                                                     \
  y = _mm256_add_ps(y,                                                         \
                    *reinterpret_cast<const __m256*>(_ps256_cephes_exp_p4));   \
  y = _mm256_mul_ps(y, x);                                                     \
  y = _mm256_add_ps(y,                                                         \
                    *reinterpret_cast<const __m256*>(_ps256_cephes_exp_p5));   \
  y = _mm256_mul_ps(y, z);                                                     \
  y = _mm256_add_ps(y, x);                                                     \
  y = _mm256_add_ps(y, one);                                                   \
  /* build 2^n */                                                              \
  imm0 = _mm256_cvttps_epi32(fx)

__m256 ExpAVX(__m256 x) {
  AVXEXP_BASE;
  // two AVX2 instructions using SSE2
  imm0 = avx2_mm256_add_epi32(imm0,
                              *reinterpret_cast<const __m256i*>(_pi256_0x7f));
  imm0 = avx2_mm256_slli_epi32(imm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}
#endif

#ifdef __AVX2__
__m256 ExpAVX2(__m256 x) {
  AVXEXP_BASE;
  // two AVX2 instructions
  imm0 = _mm256_add_epi32(imm0, *reinterpret_cast<const __m256i*>(_pi256_0x7f));
  imm0 = _mm256_slli_epi32(imm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}
#endif

}  // namespace detail
}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
