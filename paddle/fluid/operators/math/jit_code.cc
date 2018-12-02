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

#include "paddle/fluid/operators/math/jit_code.h"
#include <stddef.h>                                  // offsetof
#include "paddle/fluid/operators/math/jit_kernel.h"  // TODO(TJ): remove me

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {
namespace gen {

using namespace platform::jit;  // NOLINT

bool VXXJitCode::init(int d, int scalar_index) {
  // It's not necessary to use avx512 since it would slow down the frequency
  // and this kernel is not compute bound.
  return MayIUse(avx) && scalar_index >= 0 && scalar_index <= 2;
}

void VXXJitCode::generate() {
  // do not need push stack, and do not need save avx512reg if do not use avx512
  int offset = 0;
  if (with_relu_) {
    vxorps(ymm_zero, ymm_zero, ymm_zero);
  }
  if (scalar_index_ == 1) {
    vbroadcastss(ymm_src1, ptr[param1]);
  } else if (scalar_index_ == 2) {
    vbroadcastss(ymm_src2, ptr[param2]);
  }
  for (int i = 0; i < num_ / YMM_FLOAT_BLOCK; ++i) {
    if (scalar_index_ != 1) {
      vmovups(ymm_src1, ptr[param1 + offset]);
    }
    if (scalar_index_ != 2) {
      vmovups(ymm_src2, ptr[param2 + offset]);
    }
    if (type_ == operand_type::mul) {
      vmulps(ymm_dst, ymm_src1, ymm_src2);
    } else if (type_ == operand_type::add) {
      vaddps(ymm_dst, ymm_src1, ymm_src2);
    }
    if (with_relu_) {
      vmaxps(ymm_dst, ymm_zero, ymm_dst);
    }
    vmovups(ptr[param3 + offset], ymm_dst);
    offset += sizeof(float) * YMM_FLOAT_BLOCK;
  }
  int rest = num_ % YMM_FLOAT_BLOCK;
  while (rest > 0) {
    int block = XMM_FLOAT_BLOCK;
    if (rest >= 4) {
      block = 4;
      if (scalar_index_ != 1) {
        vmovups(xmm_src1, ptr[param1 + offset]);
      }
      if (scalar_index_ != 2) {
        vmovups(xmm_src2, ptr[param2 + offset]);
      }
    } else if (rest >= 2) {
      block = 2;
      if (scalar_index_ != 1) {
        vmovq(xmm_src1, ptr[param1 + offset]);
      }
      if (scalar_index_ != 2) {
        vmovq(xmm_src2, ptr[param2 + offset]);
      }
    } else {
      block = 1;
      if (scalar_index_ != 1) {
        vmovss(xmm_src1, ptr[param1 + offset]);
      }
      if (scalar_index_ != 2) {
        vmovss(xmm_src2, ptr[param2 + offset]);
      }
    }
    switch (type_) {
      case operand_type::mul:
        vmulps(xmm_dst, xmm_src1, xmm_src2);
        break;
      case operand_type::add:
        vaddps(xmm_dst, xmm_src1, xmm_src2);
        break;
      default:
        break;
    }
    if (with_relu_) {
      vmaxps(xmm_dst, xmm_zero, xmm_dst);
    }
    if (rest >= 4) {
      vmovups(ptr[param3 + offset], xmm_dst);
    } else if (rest >= 2) {
      vmovq(ptr[param3 + offset], xmm_dst);
    } else {
      vmovss(ptr[param3 + offset], xmm_dst);
    }
    offset += sizeof(float) * block;
    rest -= block;
  }
  ret();
}

const float exp_float_consts[] ALIGN32 = {REPEAT_8TIMES(1.f),
                                          REPEAT_8TIMES(2.f),
                                          REPEAT_8TIMES(0.5f),
                                          REPEAT_8TIMES(EXP_HIG),
                                          REPEAT_8TIMES(EXP_LOW),
                                          REPEAT_8TIMES(CEPHES_LOG2EF),
                                          REPEAT_8TIMES(CEPHES_EXP_C1),
                                          REPEAT_8TIMES(CEPHES_EXP_C2),
                                          REPEAT_8TIMES(CEPHES_EXP_P0),
                                          REPEAT_8TIMES(CEPHES_EXP_P1),
                                          REPEAT_8TIMES(CEPHES_EXP_P2),
                                          REPEAT_8TIMES(CEPHES_EXP_P3),
                                          REPEAT_8TIMES(CEPHES_EXP_P4),
                                          REPEAT_8TIMES(CEPHES_EXP_P5),
                                          REPEAT_8TIMES(EXP_MAX_INPUT),
                                          REPEAT_8TIMES(SIGMOID_THRESHOLD_MAX),
                                          REPEAT_8TIMES(SIGMOID_THRESHOLD_MIN)};

const int exp_int_0x7f[] ALIGN32 = {REPEAT_8TIMES(0x7f)};
int g_tmp_mem[16] ALIGN32 = {0};

bool VActJitCode::init(int d, operand_type type) {
  // TODO(TJ): implement avx512, avx_exp is slower than mkl when d >= 256
  return MayIUse(avx);
}

void VActJitCode::generate() {
  int offset = 0;
  for (int i = 0; i < num_ / YMM_FLOAT_BLOCK; ++i) {
    vmovups(ymm_src, ptr[param1 + offset]);
    act<ymm_t>(ymm_dst, ymm_src, type_);
    vmovups(ptr[param2 + offset], ymm_dst);
    offset += sizeof(float) * YMM_FLOAT_BLOCK;
  }
  int rest = num_ % YMM_FLOAT_BLOCK;
  while (rest > 0) {
    int block = XMM_FLOAT_BLOCK;
    if (rest >= 4) {
      block = 4;
      vmovups(xmm_src, ptr[param1 + offset]);
    } else if (rest >= 2) {
      block = 2;
      vmovq(xmm_src, ptr[param1 + offset]);
    } else {
      block = 1;
      vmovss(xmm_src, ptr[param1 + offset]);
    }
    act<xmm_t>(xmm_dst, xmm_src, type_);
    if (rest >= 4) {
      vmovups(ptr[param2 + offset], xmm_dst);
    } else if (rest >= 2) {
      vmovq(ptr[param2 + offset], xmm_dst);
    } else {
      vmovss(ptr[param2 + offset], xmm_dst);
    }
    offset += sizeof(float) * block;
    rest -= block;
  }
  ret();
}

bool LSTMJitCode::init(int d) { return MayIUse(avx) && d % 8 == 0; }

void LSTMJitCode::generate() {
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
  int d = num_ * sizeof(float);
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

bool GRUJitCode::init(int d) { return MayIUse(avx) && d % 8 == 0; }

void GRUJitCode::generate() {
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
  int d = num_ * sizeof(float);
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
}  // namespace gen
}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
