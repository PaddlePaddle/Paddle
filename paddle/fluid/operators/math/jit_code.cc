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
  xmm_t xmm_zero = xmm_t(2);
  ymm_t ymm_zero = ymm_t(2);
  if (type_ == operand_type::relu) {
    vxorps(ymm_zero, ymm_zero, ymm_zero);
  }
  int offset = 0;
  for (int i = 0; i < num_ / YMM_FLOAT_BLOCK; ++i) {
    vmovups(ymm_src, ptr[param1 + offset]);
    switch (type_) {
      case operand_type::relu:
        relu_jmm<ymm_t>(ymm_dst, ymm_src, ymm_zero);
        break;
      case operand_type::exp:
        exp_jmm<ymm_t>(ymm_dst, ymm_src, 2, 3, 4, 5);
        break;
      case operand_type::sigmoid:
        sigmoid_jmm<ymm_t>(ymm_dst, ymm_src, 2, 3, 4, 5);
        break;
      case operand_type::tanh:
        tanh_jmm<ymm_t>(ymm_dst, ymm_src, 2, 3, 4, 5);
        break;
      case operand_type::identity:
        break;
      default:
        break;
    }
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
    switch (type_) {
      case operand_type::relu:
        relu_jmm<xmm_t>(xmm_dst, xmm_src, xmm_zero);
        break;
      case operand_type::exp:
        exp_jmm<xmm_t>(xmm_dst, xmm_src, 2, 3, 4, 5);
        break;
      case operand_type::sigmoid:
        sigmoid_jmm<xmm_t>(xmm_dst, xmm_src, 2, 3, 4, 5);
        break;
      case operand_type::tanh:
        tanh_jmm<xmm_t>(xmm_dst, xmm_src, 2, 3, 4, 5);
        break;
      default:
        break;
    }
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

}  // namespace gen
}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
