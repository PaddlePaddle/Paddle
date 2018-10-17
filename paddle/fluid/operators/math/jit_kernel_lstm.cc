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
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"

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

#ifdef __AVX__
typedef enum { kSigmoid, kRelu, kTanh, kIdentity } act_type;

class AVXAct {
 public:
  virtual ~AVXAct() = default;
  virtual __m256 Compute(__m256 x) const = 0;
};

template <act_type type>
class AVXActImpl : public AVXAct {
 public:
  __m256 Compute(__m256 x) const override { PADDLE_THROW("Unkown type!"); }
};

template <>
__m256 AVXActImpl<kSigmoid>::Compute(__m256 x) const {
  __m256 ones = _mm256_set1_ps(1.0f);
  x = _mm256_max_ps(x, _mm256_set1_ps(SIGMOID_THRESHOLD_MIN));
  x = _mm256_min_ps(x, _mm256_set1_ps(SIGMOID_THRESHOLD_MAX));
  x = _mm256_sub_ps(_mm256_set1_ps(0.0f), x);
  x = detail::Exp(x);
  x = _mm256_add_ps(ones, x);
  return _mm256_div_ps(ones, x);
}

template <>
__m256 AVXActImpl<kTanh>::Compute(__m256 x) const {
  __m256 ones = _mm256_set1_ps(1.0f);
  x = _mm256_mul_ps(_mm256_set1_ps(-2.0f), x);
  x = _mm256_min_ps(x, _mm256_set1_ps(EXP_MAX_INPUT));
  x = detail::Exp(x);
  x = _mm256_add_ps(ones, x);
  x = _mm256_div_ps(_mm256_set1_ps(2.0f), x);
  return _mm256_sub_ps(x, ones);
}

template <>
__m256 AVXActImpl<kRelu>::Compute(__m256 x) const {
  return _mm256_max_ps(x, _mm256_setzero_ps());
}

template <>
__m256 AVXActImpl<kIdentity>::Compute(__m256 x) const {
  return x;
}
#endif

template <typename T>
static std::shared_ptr<const VActKernel<T>> GetActKernel(
    const std::string& type, int n) {
  if (type == "sigmoid") {
    return std::dynamic_pointer_cast<const VActKernel<T>>(
        KernelPool::Instance().template Get<VSigmoidKernel<T>>(n));
  } else if (type == "relu") {
    return std::dynamic_pointer_cast<const VActKernel<T>>(
        KernelPool::Instance().template Get<VReluKernel<T>>(n));
  } else if (type == "tanh") {
    return std::dynamic_pointer_cast<const VActKernel<T>>(
        KernelPool::Instance().template Get<VTanhKernel<T>>(n));
  } else if (type == "identity" || type == "") {
    return std::dynamic_pointer_cast<const VActKernel<T>>(
        KernelPool::Instance().template Get<VIdentityKernel<T>>(n));
  }
  PADDLE_THROW("Not support type: %s", type);
  return nullptr;
}

/* LSTM JitKernel */
template <typename T, jit::cpu_isa_t isa, jit_block>
class LSTMKernelImpl : public LSTMKernel<T> {
 public:
  explicit LSTMKernelImpl(const std::string& act_gate,
                          const std::string& act_cand,
                          const std::string& act_cell, int d)
      : LSTMKernel<T>() {
    d_ = d;
    d2_ = d * 2;
    d3_ = d * 3;
    act_gate_d3_ = GetActKernel<T>(act_gate, d3_);
    act_gate_d_ = GetActKernel<T>(act_gate, d);
    act_cand_d_ = GetActKernel<T>(act_cand, d);
    act_cell_d_ = GetActKernel<T>(act_cell, d);
    vmul_d_ = KernelPool::Instance().template Get<VMulKernel<T>>(d);
    vadd_d_ = KernelPool::Instance().template Get<VAddKernel<T>>(d);
#ifdef __AVX__
    auto GetAVXAct = [&](const std::string& type) -> std::unique_ptr<AVXAct> {
      if (type == "sigmoid") {
        return std::unique_ptr<AVXAct>(new AVXActImpl<kSigmoid>());
      } else if (type == "relu") {
        return std::unique_ptr<AVXAct>(new AVXActImpl<kRelu>());
      } else if (type == "tanh") {
        return std::unique_ptr<AVXAct>(new AVXActImpl<kTanh>());
      } else if (type == "identity" || type == "") {
        return std::unique_ptr<AVXAct>(new AVXActImpl<kIdentity>());
      }
      PADDLE_THROW("Not support type: %s", type);
    };
    avx_act_gate_ = GetAVXAct(act_gate);
    avx_act_cand_ = GetAVXAct(act_cand);
    avx_act_cell_ = GetAVXAct(act_cell);
#endif
  }

  void ComputeCtHt(T* gates, const T* ct_1, T* ct, T* ht, const T* wp_data,
                   T* checked) const override {
    // gates: W_ch, W_ih, W_fh, W_oh
    act_gate_d3_->Compute(gates + d_, gates + d_);

    /* C_t = C_t-1 * fgated + cand_gated * igated */
    act_cand_d_->Compute(gates, gates);
    vmul_d_->Compute(gates, gates + d_, gates + d_);
    vmul_d_->Compute(ct_1, gates + d2_, gates + d2_);
    vadd_d_->Compute(gates + d_, gates + d2_, ct);

    /* H_t = act_cell(C_t) * ogated */
    act_cell_d_->Compute(ct, gates + d2_);
    vmul_d_->Compute(gates + d2_, gates + d3_, ht);
  }
  void ComputeC1H1(T* gates, T* ct, T* ht, const T* wp_data) const override {
    /* C_t = igated * cgated*/
    act_gate_d_->Compute(gates + d_, gates + d_);
    act_cand_d_->Compute(gates, gates);
    vmul_d_->Compute(gates, gates + d_, ct);
    /* H_t = act_cell(C_t) * ogated */
    act_gate_d_->Compute(gates + d3_, gates + d3_);
    act_cell_d_->Compute(ct, gates + d2_);
    vmul_d_->Compute(gates + d2_, gates + d3_, ht);
  }

 private:
  int d_, d2_, d3_;
  std::shared_ptr<const VActKernel<T>> act_gate_d3_, act_gate_d_, act_cand_d_,
      act_cell_d_;
  std::shared_ptr<const VMulKernel<T>> vmul_d_;
  std::shared_ptr<const VAddKernel<T>> vadd_d_;
#ifdef __AVX__
  std::unique_ptr<const AVXAct> avx_act_gate_, avx_act_cand_, avx_act_cell_;
#endif
};

#define INTRI8_FLOAT(isa)                                                    \
  template <>                                                                \
  void LSTMKernelImpl<float, isa, kEQ8>::ComputeCtHt(                        \
      float* gates, const float* ct_1, float* ct, float* ht,                 \
      const float* wp_data, float* checked) const {                          \
    /* gates: W_ch, W_ih, W_fh, W_oh */                                      \
    __m256 c, i, f, o;                                                       \
    c = _mm256_loadu_ps(gates);                                              \
    i = _mm256_loadu_ps(gates + 8);                                          \
    f = _mm256_loadu_ps(gates + 16);                                         \
    o = _mm256_loadu_ps(gates + 24);                                         \
    /* C_t = C_t-1 * fgated + cand_gated * igated*/                          \
    c = _mm256_mul_ps(avx_act_cand_->Compute(c), avx_act_gate_->Compute(i)); \
    i = _mm256_loadu_ps(ct_1);                                               \
    f = _mm256_mul_ps(i, avx_act_gate_->Compute(f));                         \
    f = _mm256_add_ps(c, f);                                                 \
    _mm256_storeu_ps(ct, f);                                                 \
    /* H_t = act_cell(C_t) * ogated */                                       \
    o = _mm256_mul_ps(avx_act_cell_->Compute(f), avx_act_gate_->Compute(o)); \
    _mm256_storeu_ps(ht, o);                                                 \
  }

// TODO(TJ): optimize keq16

#ifdef __AVX__
INTRI8_FLOAT(jit::avx);
#endif
#ifdef __AVX2__
INTRI8_FLOAT(jit::avx2);
#endif
#ifdef __AVX512F__
INTRI8_FLOAT(jit::avx512f);
#endif

/* Peephole JitKernel */
template <typename T, jit::cpu_isa_t isa, jit_block>
class PeepholeKernelImpl : public LSTMKernel<T> {
 public:
  explicit PeepholeKernelImpl(const std::string& act_gate,
                              const std::string& act_cand,
                              const std::string& act_cell, int d)
      : LSTMKernel<T>() {
    d_ = d;
    d2_ = d * 2;
    d3_ = d * 3;
    act_gate_d_ = GetActKernel<T>(act_gate, d);
    act_cand_d_ = GetActKernel<T>(act_cand, d);
    act_cell_d_ = GetActKernel<T>(act_cell, d);
    vmul_d_ = KernelPool::Instance().template Get<VMulKernel<T>>(d);
    vadd_d_ = KernelPool::Instance().template Get<VAddKernel<T>>(d);
    vadd_d2_ = KernelPool::Instance().template Get<VAddKernel<T>>(d2_);
    act_gate_d2_ = GetActKernel<T>(act_gate, d2_);
  }

  void ComputeCtHt(T* gates, const T* ct_1, T* ct, T* ht, const T* wp_data,
                   T* checked) const override {
    /* get fgated and igated*/
    vmul_d_->Compute(wp_data, ct_1, checked);
    vmul_d_->Compute(wp_data + d_, ct_1, checked + d_);
    vadd_d2_->Compute(checked, gates + d_, gates + d_);
    act_gate_d2_->Compute(gates + d_, gates + d_);
    /* C_t = C_t-1 * fgated + cand_gated * igated*/
    act_cand_d_->Compute(gates, gates);
    vmul_d_->Compute(gates, gates + d_, gates + d_);
    vmul_d_->Compute(ct_1, gates + d2_, gates + d2_);
    vadd_d_->Compute(gates + d_, gates + d2_, ct);
    /* get ogated*/
    vmul_d_->Compute(wp_data + d2_, ct, gates + d_);
    vadd_d_->Compute(gates + d_, gates + d3_, gates + d3_);
    act_gate_d_->Compute(gates + d3_, gates + d3_);
    /* H_t = act_cell(C_t) * ogated */
    act_cell_d_->Compute(ct, gates + d2_);
    vmul_d_->Compute(gates + d2_, gates + d3_, ht);
  }

  void ComputeC1H1(T* gates, T* ct, T* ht, const T* wp_data) const override {
    /* C_t = igated * cgated*/
    act_gate_d_->Compute(gates + d_, gates + d_);
    act_cand_d_->Compute(gates, gates);
    vmul_d_->Compute(gates, gates + d_, ct);
    /* get outgated, put W_oc * C_t on igated */
    vmul_d_->Compute(wp_data + d2_, ct, gates + d_);
    vadd_d_->Compute(gates + d_, gates + d3_, gates + d3_);
    /* H_t = act_cell(C_t) * ogated */
    act_gate_d_->Compute(gates + d3_, gates + d3_);
    act_cell_d_->Compute(ct, gates + d2_);
    vmul_d_->Compute(gates + d2_, gates + d3_, ht);
  }

 private:
  int d_, d2_, d3_;
  std::shared_ptr<const VActKernel<T>> act_gate_d2_, act_gate_d_, act_cand_d_,
      act_cell_d_;
  std::shared_ptr<const VMulKernel<T>> vmul_d_;
  std::shared_ptr<const VAddKernel<T>> vadd_d_, vadd_d2_;
};

#define JITKERNEL_DECLARE_LSTM(ker_class, ker_dtype)                  \
  template <>                                                         \
  std::shared_ptr<const LSTMKernel<ker_dtype>>                        \
  KernelPool::Get<LSTMKernel<ker_dtype>, const std::string&,          \
                  const std::string&, const std::string&, int, bool>( \
      const std::string& act_gate, const std::string& act_cand,       \
      const std::string& act_cell, int d, bool use_peephole)

#define JITKERNEL_KEY_LSTM(ker_key, dtype_key)                               \
  #ker_key #dtype_key + std::to_string(d) + act_gate + act_cand + act_cell + \
                                       (use_peephole ? "p" : "n")

#define JITKERNEL_NEW_LSTM_IMPL(ker, dtype, isa, k)                    \
  if (use_peephole) {                                                  \
    p = std::dynamic_pointer_cast<ker<dtype>>(                         \
        std::make_shared<PeepholeKernelImpl<dtype, isa, k>>(           \
            act_gate, act_cand, act_cell, d));                         \
  } else {                                                             \
    p = std::dynamic_pointer_cast<ker<dtype>>(                         \
        std::make_shared<ker##Impl<dtype, isa, k>>(act_gate, act_cand, \
                                                   act_cell, d));      \
  }

REGISTER_JITKERNEL_ARGS(lstm, LSTMKernel, JITKERNEL_DECLARE_LSTM,
                        JITKERNEL_KEY_LSTM, JITKERNEL_NEW_LSTM_IMPL);

#undef INTRI8_FLOAT
#undef JITKERNEL_DECLARE_LSTM
#undef JITKERNEL_KEY_LSTM
#undef JITKERNEL_NEW_LSTM_IMPL
}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
