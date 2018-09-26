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

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {

namespace jit = platform::jit;

template <>
LSTMKernel<float>::LSTMKernel(int d, const std::string& act_gate_str,
                              const std::string& act_cand_str,
                              const std::string& act_cell_str)
    : Kernel(), d_(d) {
  d2_ = d * 2;
  d3_ = d * 3;
  if (platform::jit::MayIUse(platform::jit::avx512f)) {
    math::VecActivations<float, platform::jit::avx512f> act_functor;
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

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
