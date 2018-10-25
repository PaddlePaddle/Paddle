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
namespace jitkernel {
namespace detail {
#ifdef __AVX__
__m256 ExpAVX(__m256 x);
#endif

#ifdef __AVX2__
__m256 ExpAVX2(__m256 x);
#endif

}  // namespace detail

namespace jit = platform::jit;

#ifdef __AVX__
typedef enum { kSigmoid, kRelu, kTanh, kIdentity } act_type;

class AVXAct {
 public:
  virtual ~AVXAct() = default;
  virtual __m256 Compute(__m256 x) const = 0;
};

template <act_type type, jit::cpu_isa_t isa>
class AVXActImpl : public AVXAct {
 public:
  __m256 Compute(__m256 x) const override { PADDLE_THROW("Unkown type!"); }
};

#define AVX_SIGMOID(isa, expisa)                                 \
  template <>                                                    \
  __m256 AVXActImpl<kSigmoid, isa>::Compute(__m256 x) const {    \
    __m256 ones = _mm256_set1_ps(1.0f);                          \
    x = _mm256_max_ps(x, _mm256_set1_ps(SIGMOID_THRESHOLD_MIN)); \
    x = _mm256_min_ps(x, _mm256_set1_ps(SIGMOID_THRESHOLD_MAX)); \
    x = _mm256_sub_ps(_mm256_set1_ps(0.0f), x);                  \
    x = expisa(x);                                               \
    x = _mm256_add_ps(ones, x);                                  \
    return _mm256_div_ps(ones, x);                               \
  }

#define AVX_TANH(isa, expisa)                              \
  template <>                                              \
  __m256 AVXActImpl<kTanh, isa>::Compute(__m256 x) const { \
    __m256 ones = _mm256_set1_ps(1.0f);                    \
    x = _mm256_mul_ps(_mm256_set1_ps(-2.0f), x);           \
    x = _mm256_min_ps(x, _mm256_set1_ps(EXP_MAX_INPUT));   \
    x = expisa(x);                                         \
    x = _mm256_add_ps(ones, x);                            \
    x = _mm256_div_ps(_mm256_set1_ps(2.0f), x);            \
    return _mm256_sub_ps(x, ones);                         \
  }

#define AVX_RELU(isa)                                      \
  template <>                                              \
  __m256 AVXActImpl<kRelu, isa>::Compute(__m256 x) const { \
    return _mm256_max_ps(x, _mm256_setzero_ps());          \
  }

#define AVX_IDENTITY(isa)                                      \
  template <>                                                  \
  __m256 AVXActImpl<kIdentity, isa>::Compute(__m256 x) const { \
    return x;                                                  \
  }

#define FOR_EACH_AVX_ISA(macro_) \
  macro_(jit::avx);              \
  macro_(jit::avx2);             \
  macro_(jit::avx512f)

FOR_EACH_AVX_ISA(AVX_RELU);
FOR_EACH_AVX_ISA(AVX_IDENTITY);

AVX_SIGMOID(jit::avx, detail::ExpAVX);
AVX_TANH(jit::avx, detail::ExpAVX);

#ifdef __AVX2__
AVX_SIGMOID(jit::avx2, detail::ExpAVX2);
AVX_SIGMOID(jit::avx512f, detail::ExpAVX2);
AVX_TANH(jit::avx2, detail::ExpAVX2);
AVX_TANH(jit::avx512f, detail::ExpAVX2);
#endif

#undef FOR_EACH_AVX_ISA
#undef AVX_IDENTITY
#undef AVX_RELU
#undef AVX_TANH
#undef AVX_SIGMOID

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

#ifdef __AVX__
template <jit::cpu_isa_t isa>
static std::unique_ptr<AVXAct> GetAVXAct(const std::string& type) {
  if (type == "sigmoid") {
    return std::unique_ptr<AVXAct>(new AVXActImpl<kSigmoid, isa>());
  } else if (type == "relu") {
    return std::unique_ptr<AVXAct>(new AVXActImpl<kRelu, isa>());
  } else if (type == "tanh") {
    return std::unique_ptr<AVXAct>(new AVXActImpl<kTanh, isa>());
  } else if (type == "identity" || type == "") {
    return std::unique_ptr<AVXAct>(new AVXActImpl<kIdentity, isa>());
  }
  PADDLE_THROW("Not support type: %s", type);
  return nullptr;
}
#endif

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
  LSTMKernelImpl<float, isa, kEQ8>::LSTMKernelImpl(                          \
      const std::string& act_gate, const std::string& act_cand,              \
      const std::string& act_cell, int d)                                    \
      : LSTMKernel<float>() {                                                \
    avx_act_gate_ = GetAVXAct<isa>(act_gate);                                \
    avx_act_cand_ = GetAVXAct<isa>(act_cand);                                \
    avx_act_cell_ = GetAVXAct<isa>(act_cell);                                \
  }                                                                          \
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
  }                                                                          \
  template <>                                                                \
  void LSTMKernelImpl<float, isa, kEQ8>::ComputeC1H1(                        \
      float* gates, float* ct, float* ht, const float* wp_data) const {      \
    __m256 c, i, o;                                                          \
    c = _mm256_loadu_ps(gates);                                              \
    i = _mm256_loadu_ps(gates + 8);                                          \
    o = _mm256_loadu_ps(gates + 24);                                         \
    /* C_t = igated * cgated*/                                               \
    c = _mm256_mul_ps(avx_act_gate_->Compute(i), avx_act_cand_->Compute(c)); \
    _mm256_storeu_ps(ct, c);                                                 \
    /* H_t = act_cell(C_t) * ogated */                                       \
    o = _mm256_mul_ps(avx_act_cell_->Compute(c), avx_act_gate_->Compute(o)); \
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

/* GRU JitKernel */
template <typename T, jit::cpu_isa_t isa, jit_block>
class GRUKernelImpl : public GRUKernel<T> {
 public:
  explicit GRUKernelImpl(const std::string& act_gate,
                         const std::string& act_state, int d)
      : GRUKernel<T>() {
    d_ = d;
    d2_ = d * 2;
    act_gate_d2_ = GetActKernel<T>(act_gate, d2_);
    act_gate_d_ = GetActKernel<T>(act_gate, d);
    act_state_d_ = GetActKernel<T>(act_state, d);
    vmul_d_ = KernelPool::Instance().template Get<VMulKernel<T>>(d);
  }

  void ComputeH1(T* gates, T* ht) const override {
    act_gate_d_->Compute(gates, gates);
    act_state_d_->Compute(gates + d2_, gates + d2_);
    vmul_d_->Compute(gates, gates + d2_, ht);
  }

  void ComputeHtPart1(T* gates, const T* ht_1, T* ht) const override {
    // W: {W_update, W_reset; W_state}
    act_gate_d2_->Compute(gates, gates);
    vmul_d_->Compute(ht_1, gates + d_, ht);
  }

  void ComputeHtPart2(T* gates, const T* ht_1, T* ht) const override {
    T* y = gates + d2_;
    act_state_d_->Compute(y, y);
    // out = zt*ht~ + (1-zt)*ht_1
    for (int i = 0; i < d_; ++i) {
      ht[i] = gates[i] * y[i] + (static_cast<T>(1) - gates[i]) * ht_1[i];
    }
  }

 private:
  int d_, d2_;
  std::shared_ptr<const VActKernel<T>> act_gate_d2_, act_gate_d_, act_state_d_;
  std::shared_ptr<const VMulKernel<T>> vmul_d_;
#ifdef __AVX__
  std::unique_ptr<const AVXAct> avx_act_gate_, avx_act_state_;
#endif
};

#define INTRI8_FLOAT(isa)                                                     \
  template <>                                                                 \
  GRUKernelImpl<float, isa, kEQ8>::GRUKernelImpl(                             \
      const std::string& act_gate, const std::string& act_state, int d)       \
      : GRUKernel<float>() {                                                  \
    avx_act_gate_ = GetAVXAct<isa>(act_gate);                                 \
    avx_act_state_ = GetAVXAct<isa>(act_state);                               \
  }                                                                           \
  template <>                                                                 \
  void GRUKernelImpl<float, isa, kEQ8>::ComputeH1(float* gates, float* ht)    \
      const {                                                                 \
    __m256 u, s;                                                              \
    /* W: {W_update, W_reset; W_state} */                                     \
    u = _mm256_loadu_ps(gates);                                               \
    s = _mm256_loadu_ps(gates + 16);                                          \
    s = _mm256_mul_ps(avx_act_gate_->Compute(u), avx_act_state_->Compute(s)); \
    _mm256_storeu_ps(ht, s);                                                  \
  }                                                                           \
  template <>                                                                 \
  void GRUKernelImpl<float, isa, kEQ8>::ComputeHtPart1(                       \
      float* gates, const float* ht_1, float* ht) const {                     \
    /* not exactly equal the any implementation */                            \
    __m256 r, ht0;                                                            \
    r = _mm256_loadu_ps(gates + 8);                                           \
    ht0 = _mm256_loadu_ps(ht_1);                                              \
    r = _mm256_mul_ps(avx_act_gate_->Compute(r), ht0);                        \
    _mm256_storeu_ps(ht, r);                                                  \
  }                                                                           \
  template <>                                                                 \
  void GRUKernelImpl<float, isa, kEQ8>::ComputeHtPart2(                       \
      float* gates, const float* ht_1, float* ht) const {                     \
    /* not exactly equal the any implementation */                            \
    __m256 u, s, ht0;                                                         \
    u = _mm256_loadu_ps(gates);                                               \
    s = _mm256_loadu_ps(gates + 16);                                          \
    ht0 = _mm256_loadu_ps(ht_1);                                              \
    u = avx_act_gate_->Compute(u);                                            \
    s = _mm256_mul_ps(u, avx_act_state_->Compute(s));                         \
    u = _mm256_sub_ps(_mm256_set1_ps(1.f), u);                                \
    u = _mm256_mul_ps(u, ht0);                                                \
    u = _mm256_add_ps(s, u);                                                  \
    _mm256_storeu_ps(ht, u);                                                  \
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

#define JITKERNEL_DECLARE_GRU(ker_class, ker_dtype)                       \
  template <>                                                             \
  std::shared_ptr<const GRUKernel<ker_dtype>> KernelPool::Get<            \
      GRUKernel<ker_dtype>, const std::string&, const std::string&, int>( \
      const std::string& act_gate, const std::string& act_state, int d)

#define JITKERNEL_KEY_GRU(ker_key, dtype_key) \
  #ker_key #dtype_key + std::to_string(d) + act_gate + act_state

#define JITKERNEL_NEW_GRU_IMPL(ker, dtype, isa, k) \
  p = std::dynamic_pointer_cast<ker<dtype>>(       \
      std::make_shared<ker##Impl<dtype, isa, k>>(act_gate, act_state, d));

REGISTER_JITKERNEL_ARGS(gru, GRUKernel, JITKERNEL_DECLARE_GRU,
                        JITKERNEL_KEY_GRU, JITKERNEL_NEW_GRU_IMPL);

#undef INTRI8_FLOAT
#undef JITKERNEL_NEW_GRU_IMPL
#undef JITKERNEL_KEY_GRU
#undef JITKERNEL_DECLARE_GRU
}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
