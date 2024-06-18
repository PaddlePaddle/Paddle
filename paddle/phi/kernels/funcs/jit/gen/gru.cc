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

#include "paddle/phi/kernels/funcs/jit/gen/gru.h"

#include <cstddef>  // offsetof

#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/phi/kernels/funcs/jit/macro.h"
#include "paddle/phi/kernels/funcs/jit/registry.h"

namespace phi::jit::gen {

void GRUJitCode::genCode() {
  reg64_t reg_ptr_gates = rax;
  reg64_t reg_ptr_ht_1 = r9;
  reg64_t reg_ptr_ht = r10;
  mov(reg_ptr_gates, ptr[param1 + offsetof(gru_t, gates)]);
  mov(reg_ptr_ht_1, ptr[param1 + offsetof(gru_t, ht_1)]);
  mov(reg_ptr_ht, ptr[param1 + offsetof(gru_t, ht)]);
  ymm_t ymm_one = ymm_t(0);

  if (id_ == 2) {
    reg64_t reg_ptr_tmp = r11;
    mov(reg_ptr_tmp, reinterpret_cast<size_t>(exp_float_consts));
    vmovaps(ymm_one, ptr[reg_ptr_tmp + OFFSET_EXP_ONE]);
  }
  int offset = 0;
  int d = num_ * sizeof(float);  // NOLINT
  for (int i = 0; i < num_ / YMM_FLOAT_BLOCK; ++i) {
    ymm_t ymm_u = ymm_t(1);
    ymm_t ymm_r = ymm_t(2);
    ymm_t ymm_s = ymm_t(3);
    ymm_t ymm_ht_1 = ymm_t(4);
    // W: {W_update, W_reset; W_state}
    if (id_ == 0 || id_ == 2) {
      vmovups(ymm_u, ptr[reg_ptr_gates + offset]);
      vmovups(ymm_s, ptr[reg_ptr_gates + offset + 2 * d]);
    }
    if (id_ == 1) {
      vmovups(ymm_r, ptr[reg_ptr_gates + offset + d]);
    }
    if (id_ == 1 || id_ == 2) {
      vmovups(ymm_ht_1, ptr[reg_ptr_ht_1 + offset]);
    }

    if (id_ == 0) {
      // ht = act_gate(u) * act_cand(s)
      act<ymm_t>(ymm_u, ymm_u, act_gate_);
      act<ymm_t>(ymm_s, ymm_s, act_cand_);
      vmulps(ymm_s, ymm_s, ymm_u);
      vmovups(ptr[reg_ptr_ht + offset], ymm_s);
    } else if (id_ == 1) {
      // ht = act_gate(r) * ht_1
      act<ymm_t>(ymm_r, ymm_r, act_gate_);
      vmulps(ymm_r, ymm_r, ymm_ht_1);
      vmovups(ptr[reg_ptr_ht + offset], ymm_r);
    } else if (id_ == 2) {
      // ht = act_gate(u) * act_cand(s) + (1-act_gate(u)) * ht_1
      ymm_t ymm_one_inner = ymm_t(ymm_one.getIdx());
      act<ymm_t>(ymm_u, ymm_u, act_gate_);
      act<ymm_t>(ymm_s, ymm_s, act_cand_);
      vmulps(ymm_s, ymm_s, ymm_u);
      vsubps(ymm_u, ymm_one_inner, ymm_u);
      vmulps(ymm_u, ymm_ht_1, ymm_u);
      vaddps(ymm_u, ymm_s, ymm_u);
      vmovups(ptr[reg_ptr_ht + offset], ymm_u);
    }
    offset += sizeof(float) * YMM_FLOAT_BLOCK;
  }
  ret();
}

#define DECLARE_GRU_CREATOR(name)                                    \
  class name##Creator : public JitCodeCreator<gru_attr_t> {          \
   public:                                                           \
    /* TODO(TJ): enable more */                                      \
    bool CanBeUsed(const gru_attr_t& attr) const override {          \
      return phi::backends::cpu::MayIUse(phi::backends::cpu::avx) && \
             attr.d % 8 == 0;                                        \
    }                                                                \
    size_t CodeSize(const gru_attr_t& attr) const override {         \
      return 96 + attr.d / YMM_FLOAT_BLOCK * 96 * 2 * 8;             \
    }                                                                \
    std::unique_ptr<GenBase> CreateJitCode(                          \
        const gru_attr_t& attr) const override {                     \
      return make_unique<name##JitCode>(attr, CodeSize(attr));       \
    }                                                                \
  }

DECLARE_GRU_CREATOR(GRUH1);
DECLARE_GRU_CREATOR(GRUHtPart1);
DECLARE_GRU_CREATOR(GRUHtPart2);

#undef DECLARE_GRU_CREATOR

}  // namespace phi::jit::gen

namespace gen = phi::jit::gen;

REGISTER_JITKERNEL_GEN(kGRUH1, gen::GRUH1Creator);
REGISTER_JITKERNEL_GEN(kGRUHtPart1, gen::GRUHtPart1Creator);
REGISTER_JITKERNEL_GEN(kGRUHtPart2, gen::GRUHtPart2Creator);
