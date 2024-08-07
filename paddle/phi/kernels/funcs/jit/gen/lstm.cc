/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include "paddle/phi/kernels/funcs/jit/gen/lstm.h"

#include <cstddef>  // offsetof

#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/phi/kernels/funcs/jit/macro.h"
#include "paddle/phi/kernels/funcs/jit/registry.h"

namespace phi::jit::gen {

void LSTMJitCode::genCode() {
  if (use_peephole_) {
    preCode();
  }
  reg64_t reg_ptr_gates = rax;
  reg64_t reg_ptr_ct_1 = r9;
  reg64_t reg_ptr_ct = r10;
  reg64_t reg_ptr_ht = r11;
  reg64_t reg_ptr_wp = r12;
  mov(reg_ptr_gates, ptr[param1 + offsetof(lstm_t, gates)]);
  mov(reg_ptr_ct_1, ptr[param1 + offsetof(lstm_t, ct_1)]);
  mov(reg_ptr_ct, ptr[param1 + offsetof(lstm_t, ct)]);
  mov(reg_ptr_ht, ptr[param1 + offsetof(lstm_t, ht)]);
  if (use_peephole_) {
    mov(reg_ptr_wp, ptr[param1 + offsetof(lstm_t, wp)]);
  }

  int offset = 0;
  int d = num_ * sizeof(float);  // NOLINT
  for (int i = 0; i < num_ / YMM_FLOAT_BLOCK; ++i) {
    /* gates: W_ch, W_ih, W_fh, W_oh */
    ymm_t ymm_c = ymm_t(0);
    ymm_t ymm_i = ymm_t(1);
    ymm_t ymm_f = ymm_t(2);
    ymm_t ymm_o = ymm_t(3);
    ymm_t ymm_ct_1 = ymm_t(4);
    ymm_t ymm_wp0 = ymm_t(5);
    ymm_t ymm_wp1 = ymm_t(6);
    ymm_t ymm_wp2 = ymm_t(7);
    vmovups(ymm_c, ptr[reg_ptr_gates + offset]);
    vmovups(ymm_i, ptr[reg_ptr_gates + offset + d]);
    vmovups(ymm_f, ptr[reg_ptr_gates + offset + 2 * d]);
    vmovups(ymm_o, ptr[reg_ptr_gates + offset + 3 * d]);
    if (!compute_c1h1_) {
      vmovups(ymm_ct_1, ptr[reg_ptr_ct_1 + offset]);
    }
    if (use_peephole_) {
      vmovups(ymm_wp0, ptr[reg_ptr_wp + offset]);
      vmovups(ymm_wp1, ptr[reg_ptr_wp + offset + d]);
      vmovups(ymm_wp2, ptr[reg_ptr_wp + offset + 2 * d]);
    }
    /* C_t = act_cand(c) * act_gate(i) + C_t-1 * act_gate(f) */
    // act_cand(c)
    act<ymm_t>(ymm_c, ymm_c, act_cand_);
    // act_gate(i) or act_gate(ct_1 * wp0 + i)
    if (!compute_c1h1_ && use_peephole_) {
      vmulps(ymm_wp0, ymm_ct_1, ymm_wp0);
      vaddps(ymm_i, ymm_i, ymm_wp0);
    }
    act<ymm_t>(ymm_i, ymm_i, act_gate_);
    vmulps(ymm_c, ymm_c, ymm_i);
    if (!compute_c1h1_) {
      // act_gate(f) or act_gate(ct_1 * wp1 + f)
      if (use_peephole_) {
        vmulps(ymm_wp1, ymm_ct_1, ymm_wp1);
        vaddps(ymm_f, ymm_f, ymm_wp1);
      }
      act<ymm_t>(ymm_f, ymm_f, act_gate_);
      // ct
      vmulps(ymm_f, ymm_f, ymm_ct_1);
      vaddps(ymm_f, ymm_f, ymm_c);
    }
    /* H_t = act_cell(C_t) * act_gate(o) */
    // act_cell(C_t)
    ymm_t ymm_ct = compute_c1h1_ ? ymm_c : ymm_f;
    ymm_t ymm_tmp = ymm_i;
    act<ymm_t>(ymm_tmp, ymm_ct, act_cell_);
    // act_gate(o) or act_gate(ct * wp2 + o)
    if (use_peephole_) {
      vmulps(ymm_wp2, ymm_ct, ymm_wp2);
      vaddps(ymm_o, ymm_o, ymm_wp2);
    }
    act<ymm_t>(ymm_o, ymm_o, act_gate_);
    // ht
    vmulps(ymm_o, ymm_o, ymm_tmp);
    // save ct and ht
    vmovups(ptr[reg_ptr_ct + offset], ymm_ct);
    vmovups(ptr[reg_ptr_ht + offset], ymm_o);
    offset += sizeof(float) * YMM_FLOAT_BLOCK;
  }

  if (use_peephole_) {
    postCode();
  } else {
    ret();
  }
}

#define DECLARE_LSTM_CREATOR(name)                                   \
  class name##Creator : public JitCodeCreator<lstm_attr_t> {         \
   public:                                                           \
    /* TODO(TJ): enable more */                                      \
    bool CanBeUsed(const lstm_attr_t& attr) const override {         \
      return phi::backends::cpu::MayIUse(phi::backends::cpu::avx) && \
             attr.d % 8 == 0;                                        \
    }                                                                \
    size_t CodeSize(const lstm_attr_t& attr) const override {        \
      return 96 + attr.d / YMM_FLOAT_BLOCK * 90 * 4 * 8;             \
    }                                                                \
    std::unique_ptr<GenBase> CreateJitCode(                          \
        const lstm_attr_t& attr) const override {                    \
      return make_unique<name##JitCode>(attr, CodeSize(attr));       \
    }                                                                \
  }

DECLARE_LSTM_CREATOR(LSTMCtHt);
DECLARE_LSTM_CREATOR(LSTMC1H1);

#undef DECLARE_LSTM_CREATOR

}  // namespace phi::jit::gen

namespace gen = phi::jit::gen;

REGISTER_JITKERNEL_GEN(kLSTMCtHt, gen::LSTMCtHtCreator);
REGISTER_JITKERNEL_GEN(kLSTMC1H1, gen::LSTMC1H1Creator);
