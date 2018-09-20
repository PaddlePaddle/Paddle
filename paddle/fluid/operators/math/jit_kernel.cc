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
#include <functional>
#include <string>
#include "paddle/fluid/operators/math/cpu_vec.h"

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

KernelPool& KernelPool::Instance() {
  static KernelPool g_jit_kernels;
  return g_jit_kernels;
}
#define SEARCH_BLOCK(src, t, isa)       \
  if (d < AVX_FLOAT_BLOCK) {            \
    Compute = src<t, isa, kLT8>;        \
  } else if (d == AVX_FLOAT_BLOCK) {    \
    Compute = src<t, isa, kEQ8>;        \
  } else if (d == AVX512_FLOAT_BLOCK) { \
    Compute = src<t, isa, kEQ16>;       \
  } else {                              \
    Compute = src<t, isa, kGT16>;       \
  }

#define SEARCH_ISA_BLOCK(src, t)              \
  if (jit::MayIUse(jit::avx512_common)) {     \
    SEARCH_BLOCK(src, t, jit::avx512_common); \
  } else if (jit::MayIUse(jit::avx2)) {       \
    SEARCH_BLOCK(src, t, jit::avx2);          \
  } else if (jit::MayIUse(jit::avx)) {        \
    SEARCH_BLOCK(src, t, jit::avx);           \
  } else {                                    \
    SEARCH_BLOCK(src, t, jit::isa_any);       \
  }

#define FOR_EACH_BLOCK(macro_, isa) \
  macro_(isa, kLT8) macro_(isa, kEQ8) macro_(isa, kEQ16) macro_(isa, kGT16)

#define FOR_EACH_ISA_BLOCK(macro_)           \
  FOR_EACH_BLOCK(macro_, jit::avx512_common) \
  FOR_EACH_BLOCK(macro_, jit::avx2)          \
  FOR_EACH_BLOCK(macro_, jit::avx)           \
  FOR_EACH_BLOCK(macro_, jit::any)

#define VMUL_ANY                \
  for (int i = 0; i < n; ++i) { \
    z[i] = x[i] * y[i];         \
  }

template <typename T, platform::jit::cpu_isa_t isa, jit_block>
static void VMulCompute(const int n, const T* x, const T* y, T* z) {
  VMUL_ANY
}

#ifdef PADDLE_USE_MKLML
#define DEFINE_VMUL_COMPUTE_FLOAT(isa, block)                             \
  template <>                                                             \
  static void VMulCompute<float, isa, block>(const int n, const float* x, \
                                             const float* y, float* z) {  \
    platform::dynload::vsMul(n, x, y, z);                                 \
  }

#define DEFINE_VMUL_COMPUTE_DOUBLE(isa, block)                              \
  template <>                                                               \
  static void VMulCompute<double, isa, block>(const int n, const double* x, \
                                              const double* y, float* z) {  \
    platform::dynload::vdMul(n, x, y, z);                                   \
  }

FOR_EACH_ISA_BLOCK(DEFINE_VMUL_COMPUTE_FLOAT)
FOR_EACH_ISA_BLOCK(DEFINE_VMUL_COMPUTE_DOUBLE)
// TODO(TJ): add EQ8
#endif

#undef DEFINE_VMUL_COMPUTE_FLOAT
#undef DEFINE_VMUL_COMPUTE_DOUBLE
#undef VMUL_ANY

template <>
VMulKernel<float>::VMulKernel(int d) {
  SEARCH_ISA_BLOCK(VMulCompute, float);
}

template <>
VMulKernel<double>::VMulKernel(int d) {
  SEARCH_ISA_BLOCK(VMulCompute, double);
}

template <>
const std::shared_ptr<VMulKernel<float>> KernelPool::Get<VMulKernel<float>>(
    int d) {
  std::string key = "f" + std::to_string(d);
  if (kers_.find(key) == kers_.end()) {
    auto p = std::make_shared<VMulKernel<float>>(d);
    kers_.insert({key, std::dynamic_pointer_cast<Kernel>(p)});
    return p;
  }
  return std::dynamic_pointer_cast<VMulKernel<float>>(kers_.at(key));
}

template <>
const std::shared_ptr<VMulKernel<double>> KernelPool::Get<VMulKernel<double>>(
    int d) {
  std::string key = "d" + std::to_string(d);
  if (kers_.find(key) == kers_.end()) {
    auto p = std::make_shared<VMulKernel<double>>(d);
    kers_.insert({key, std::dynamic_pointer_cast<Kernel>(p)});
    return p;
  }
  return std::dynamic_pointer_cast<VMulKernel<double>>(kers_.at(key));
}

template <>
LSTMKernel<float>::LSTMKernel(int d, const std::string& act_gate_str,
                              const std::string& act_cand_str,
                              const std::string& act_cell_str)
    : Kernel(), d_(d) {
  d2_ = d * 2;
  d3_ = d * 3;
  if (platform::jit::MayIUse(platform::jit::avx512_common)) {
    math::VecActivations<float, platform::jit::avx512_common> act_functor;
    act_gate_ = act_functor(act_gate_str);
    act_cell_ = act_functor(act_cell_str);
    act_cand_ = act_functor(act_cand_str);
  } else if (platform::jit::MayIUse(platform::jit::avx2)) {
    math::VecActivations<float, platform::jit::avx2> act_functor;
    act_gate_ = act_functor(act_gate_str);
    act_cell_ = act_functor(act_cell_str);
    act_cand_ = act_functor(act_cand_str);
  } else if (platform::jit::MayIUse(platform::jit::avx)) {
    math::VecActivations<float, platform::jit::avx> act_functor;
    act_gate_ = act_functor(act_gate_str);
    act_cell_ = act_functor(act_cell_str);
    act_cand_ = act_functor(act_cand_str);
    //   ComputeCtHt = [&](float*gates,const float*ct_1,float*ct, float*ht) {
    // // gates: W_ch, W_ih, W_fh, W_oh
    // act_gate(d3_, gates + d_, gates + d_);

    // /* C_t = C_t-1 * fgated + cand_gated * igated */
    // act_cand(d_, gates, gates);
    // blas.VMUL(d_, gates, gates + d_, gates + d_);
    // blas.VMUL(d_, ct_1, gates + d2_, gates + d2_);
    // blas.VADD(d_, gates + d_, gates + d2_, ct);

    // /* H_t = act_cell(C_t) * ogated */
    // act_cell(d_, ct, gates + d2_);
    // blas.VMUL(d_, gates + d2_, gates + d3_, ht)
    // GET_Ct(ct_1, gates, ct);
    // GET_Ht(ct, gates, ht);
    //   };
  } else {
    math::VecActivations<float, platform::jit::isa_any> act_functor;
    act_gate_ = act_functor(act_gate_str);
    act_cell_ = act_functor(act_cell_str);
    act_cand_ = act_functor(act_cand_str);
  }
}

template <>
const std::shared_ptr<LSTMKernel<float>>
KernelPool::Get<LSTMKernel<float>, int, const std::string&, const std::string&,
                const std::string&>(int d, const std::string& act_gate,
                                    const std::string& act_cand,
                                    const std::string& act_cell) {
  std::string key = "f" + std::to_string(d) + act_gate + act_cand + act_cell;
  if (kers_.find(key) == kers_.end()) {
    auto p =
        std::make_shared<LSTMKernel<float>>(d, act_gate, act_cand, act_cell);
    kers_.insert({key, std::dynamic_pointer_cast<Kernel>(p)});
    return p;
  }
  return std::dynamic_pointer_cast<LSTMKernel<float>>(kers_.at(key));
}

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
