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
#include "paddle/fluid/operators/math/jit_kernel.h"
#include "paddle/fluid/platform/cpu_info.h"

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
  for (int i = 0; i < num_ / AVX_FLOAT_BLOCK; ++i) {
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
    offset += sizeof(float) * AVX_FLOAT_BLOCK;
  }
  int rest = num_ % AVX_FLOAT_BLOCK;
  if (rest >= 4) {
    if (scalar_index_ != 1) {
      vmovups(xmm_src1, ptr[param1 + offset]);
    }
    if (scalar_index_ != 2) {
      vmovups(xmm_src2, ptr[param2 + offset]);
    }
    if (type_ == operand_type::mul) {
      vmulps(xmm_dst, xmm_src1, xmm_src2);
    } else if (type_ == operand_type::add) {
      vaddps(xmm_dst, xmm_src1, xmm_src2);
    }
    if (with_relu_) {
      vmaxps(xmm_dst, xmm_zero, xmm_dst);
    }
    vmovups(ptr[param3 + offset], xmm_dst);
    offset += sizeof(float) * 4;
    rest -= 4;
  }
  if (rest >= 2) {
    if (scalar_index_ != 1) {
      vmovups(xmm_src1, ptr[param1 + offset]);
    }
    if (scalar_index_ != 2) {
      vmovups(xmm_src2, ptr[param2 + offset]);
    }
    if (type_ == operand_type::mul) {
      vmulps(xmm_dst, xmm_src1, xmm_src2);
    } else if (type_ == operand_type::add) {
      vaddps(xmm_dst, xmm_src1, xmm_src2);
    }
    if (with_relu_) {
      vmaxps(xmm_dst, xmm_zero, xmm_dst);
    }
    vmovq(ptr[param3 + offset], xmm_dst);
    offset += sizeof(float) * 2;
    rest -= 2;
  }
  if (rest > 0) {
    if (scalar_index_ != 1) {
      vmovups(xmm_src1, ptr[param1 + offset]);
    }
    if (scalar_index_ != 2) {
      vmovups(xmm_src2, ptr[param2 + offset]);
    }
    if (type_ == operand_type::mul) {
      vmulss(xmm_dst, xmm_src1, xmm_src2);
    } else if (type_ == operand_type::add) {
      vaddss(xmm_dst, xmm_src1, xmm_src2);
    }
    if (with_relu_) {
      vmaxps(xmm_dst, xmm_zero, xmm_dst);
    }
    vmovss(ptr[param3 + offset], xmm_dst);
  }
  ret();
}

bool ReluJitCode::init(int d) { return MayIUse(avx); }

void ReluJitCode::generate() {
  int offset = 0;
  vxorps(ymm_zero, ymm_zero, ymm_zero);
  for (int i = 0; i < num_ / AVX_FLOAT_BLOCK; ++i) {
    vmovups(ymm_src, ptr[param1 + offset]);
    vmaxps(ymm_dst, ymm_zero, ymm_src);
    vmovups(ptr[param2 + offset], ymm_dst);
    offset += sizeof(float) * AVX_FLOAT_BLOCK;
  }
  int rest = num_ % AVX_FLOAT_BLOCK;
  if (rest >= 4) {
    vmovups(xmm_src, ptr[param1 + offset]);
    vmaxps(xmm_dst, xmm_zero, xmm_src);
    vmovups(ptr[param2 + offset], xmm_dst);
    offset += sizeof(float) * 4;
    rest -= 4;
  }
  if (rest >= 2) {
    vmovups(xmm_src, ptr[param1 + offset]);
    vmaxps(xmm_dst, xmm_zero, xmm_src);
    vmovq(ptr[param2 + offset], xmm_dst);
    offset += sizeof(float) * 2;
    rest -= 2;
  }
  if (rest > 0) {
    vmovups(xmm_src, ptr[param1 + offset]);
    vmaxps(xmm_dst, xmm_zero, xmm_src);
    vmovss(ptr[param2 + offset], xmm_dst);
  }
  ret();
}

#define ALIGN32 __attribute__((aligned(32)))
#define EXP_HIG 88.3762626647949f
#define EXP_LOW -88.3762626647949f
#define CEPHES_LOG2EF 1.44269504088896341
#define CEPHES_EXP_C1 0.693359375
#define CEPHES_EXP_C2 -2.12194440e-4
#define CEPHES_EXP_P0 1.9875691500E-4
#define CEPHES_EXP_P1 1.3981999507E-3
#define CEPHES_EXP_P2 8.3334519073E-3
#define CEPHES_EXP_P3 4.1665795894E-2
#define CEPHES_EXP_P4 1.6666665459E-1
#define CEPHES_EXP_P5 5.0000001201E-1

#define REPEAT_8TIMES(val) val, val, val, val, val, val, val, val

#define OFFSET_EXP_ONE 0 * AVX_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_TWO 1 * AVX_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_0P5 2 * AVX_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_HIG 3 * AVX_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_LOW 4 * AVX_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_LOG2EF 5 * AVX_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_C1 6 * AVX_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_C2 7 * AVX_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_P0 8 * AVX_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_P1 9 * AVX_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_P2 10 * AVX_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_P3 11 * AVX_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_P4 12 * AVX_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_P5 13 * AVX_FLOAT_BLOCK * sizeof(float)
#define OFFSET_EXP_MAX_INPUT 14 * AVX_FLOAT_BLOCK * sizeof(float)
#define OFFSET_SIGMOID_MAX 15 * AVX_FLOAT_BLOCK * sizeof(float)
#define OFFSET_SIGMOID_MIN 16 * AVX_FLOAT_BLOCK * sizeof(float)

static const float exp_float_consts[] ALIGN32 = {
    REPEAT_8TIMES(1.f),
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

static const int exp_int_0x7f[] ALIGN32 = {REPEAT_8TIMES(0x7f)};
static int g_tmp_mem[16] ALIGN32 = {0};

bool VExpJitCode::init(int d) {
  return MayIUse(avx) && d == 8;  // only 8 yet
}

void VExpJitCode::exp_ymm(ymm_t& ymm_src, ymm_t& ymm_dst) {
  // use reg rax and ymm 2~5
  reg64_t reg_ptr_global = rax;
  ymm_t ymm_fx = ymm_t(2);
  ymm_t ymm_fy = ymm_t(3);
  ymm_t ymm_mask = ymm_t(4);
  ymm_t ymm_tmp = ymm_t(5);
  assert(ymm_src.getIdx() != ymm_dst.getIdx());  // TODO(TJ): use enfore
  push(reg_ptr_global);
  mov(reg_ptr_global, reinterpret_cast<size_t>(exp_float_consts));
  vmovaps(ymm_tmp, ptr[reg_ptr_global + OFFSET_EXP_HIG]);
  vminps(ymm_src, ymm_src, ymm_tmp);
  vmovaps(ymm_tmp, ptr[reg_ptr_global + OFFSET_EXP_LOW]);
  vmaxps(ymm_src, ymm_src, ymm_tmp);
  // express exp(x) as exp(g + n*log(2))
  vmovaps(ymm_tmp, ptr[reg_ptr_global + OFFSET_EXP_LOG2EF]);
  vmulps(ymm_fx, ymm_src, ymm_tmp);
  vmovaps(ymm_tmp, ptr[reg_ptr_global + OFFSET_EXP_0P5]);
  vaddps(ymm_fx, ymm_fx, ymm_tmp);
  vroundps(ymm_fy, ymm_fx, 0x01);
  // if greater, substract 1
  vcmpgtps(ymm_mask, ymm_fy, ymm_fx);
  vmovaps(ymm_tmp, ptr[reg_ptr_global]);
  vandps(ymm_mask, ymm_mask, ymm_tmp);
  vsubps(ymm_fx, ymm_fy, ymm_mask);
  vmovaps(ymm_tmp, ptr[reg_ptr_global + OFFSET_EXP_C1]);
  vmulps(ymm_fy, ymm_fx, ymm_tmp);
  vmovaps(ymm_tmp, ptr[reg_ptr_global + OFFSET_EXP_C2]);
  ymm_t ymm_z = ymm_t(ymm_mask.getIdx());
  vmulps(ymm_z, ymm_fx, ymm_tmp);
  vsubps(ymm_src, ymm_src, ymm_fy);
  vsubps(ymm_src, ymm_src, ymm_z);
  vmulps(ymm_z, ymm_src, ymm_src);
  vmovaps(ymm_tmp, ptr[reg_ptr_global + OFFSET_EXP_P0]);
  vmulps(ymm_dst, ymm_src, ymm_tmp);
  for (size_t i = OFFSET_EXP_P1; i < OFFSET_EXP_P5;
       i += (AVX_FLOAT_BLOCK * sizeof(float))) {
    vmovaps(ymm_tmp, ptr[reg_ptr_global + i]);  // P1~P4
    vaddps(ymm_dst, ymm_dst, ymm_tmp);
    vmulps(ymm_dst, ymm_dst, ymm_src);
  }
  vmovaps(ymm_tmp, ptr[reg_ptr_global + OFFSET_EXP_P5]);
  vaddps(ymm_dst, ymm_dst, ymm_tmp);
  vmulps(ymm_dst, ymm_dst, ymm_z);
  vaddps(ymm_dst, ymm_dst, ymm_src);
  vmovaps(ymm_tmp, ptr[reg_ptr_global]);
  vaddps(ymm_dst, ymm_dst, ymm_tmp);
  // build 2^n
  ymm_t ymm_int = ymm_fx;
  vcvttps2dq(ymm_int, ymm_fx);
  mov(reg_ptr_global, reinterpret_cast<size_t>(exp_int_0x7f));
  vmovdqa(ymm_tmp, ptr[reg_ptr_global]);
  if (MayIUse(avx2)) {
    vpaddd(ymm_int, ymm_int, ymm_tmp);
    vpslld(ymm_int, ymm_int, 23);
  } else if (MayIUse(avx)) {
    xmm_t xtmp1 = xmm_t(ymm_int.getIdx());
    xmm_t xtmp2 = xmm_t(ymm_tmp.getIdx());
    reg64_t reg_ptr_tmp = reg_ptr_global;
    mov(reg_ptr_tmp, reinterpret_cast<size_t>(g_tmp_mem));
    vmovdqa(ptr[reg_ptr_tmp], ymm_int);
    vmovdqa(ptr[reg_ptr_tmp + AVX_FLOAT_BLOCK * sizeof(float)], ymm_tmp);
    vpaddd(xtmp1, xtmp1, xtmp2);
    vpslld(xtmp1, xtmp1, 23);
    vmovdqa(ptr[reg_ptr_tmp], xtmp1);
    // next 128bits
    vmovdqa(xtmp1, ptr[reg_ptr_tmp + 4 /*xmm float block*/ * sizeof(float)]);
    vmovdqa(xtmp2,
            ptr[reg_ptr_tmp +
                (AVX_FLOAT_BLOCK + 4 /*xmm float block*/) * sizeof(float)]);
    vpaddd(xtmp1, xtmp1, xtmp2);
    vpslld(xtmp1, xtmp1, 23);
    vmovdqa(ptr[reg_ptr_tmp + 4 /*xmm float block*/ * sizeof(float)], xtmp1);
    // load out
    vmovdqa(ymm_int, ptr[reg_ptr_tmp]);
  }
  vmulps(ymm_dst, ymm_dst, ymm_int);
  pop(reg_ptr_global);
}

void VExpJitCode::generate() {
  int offset = 0;
  vmovups(ymm_src, ptr[param1 + offset]);
  exp_ymm(ymm_src, ymm_dst);
  vmovups(ptr[param2 + offset], ymm_dst);
  ret();
}

bool VSigmoidJitCode::init(int d) {
  return MayIUse(avx) && d == 8;  // only 8 yet
}

void VSigmoidJitCode::sigmoid_ymm(ymm_t& ymm_src, ymm_t& ymm_dst) {
  // use ymm2
  reg64_t reg_ptr_global = rax;
  ymm_t ymm_tmp = ymm_t(2);
  push(reg_ptr_global);
  mov(reg_ptr_global, reinterpret_cast<size_t>(exp_float_consts));
  vmovaps(ymm_tmp, ptr[reg_ptr_global + OFFSET_SIGMOID_MAX]);
  vminps(ymm_src, ymm_src, ymm_tmp);
  vmovaps(ymm_tmp, ptr[reg_ptr_global + OFFSET_SIGMOID_MIN]);
  vmaxps(ymm_src, ymm_src, ymm_tmp);
  vxorps(ymm_tmp, ymm_tmp, ymm_tmp);
  vsubps(ymm_src, ymm_tmp, ymm_src);
  exp_ymm(ymm_src, ymm_dst);
  vmovaps(ymm_tmp, ptr[reg_ptr_global + OFFSET_EXP_ONE]);
  vaddps(ymm_dst, ymm_dst, ymm_tmp);
  vdivps(ymm_dst, ymm_tmp, ymm_dst);
  pop(reg_ptr_global);
}

void VSigmoidJitCode::generate() {
  int offset = 0;
  vmovups(ymm_src, ptr[param1 + offset]);
  sigmoid_ymm(ymm_src, ymm_dst);
  vmovups(ptr[param2 + offset], ymm_dst);
  ret();
}

bool VTanhJitCode::init(int d) {
  return MayIUse(avx) && d == 8;  // only 8 yet
}

void VTanhJitCode::vtanh_ymm(ymm_t& ymm_src, ymm_t& ymm_dst) {
  // y = 2 / (1 + e^(-2x)) - 1
  // use ymm2, ymm3
  reg64_t reg_ptr_global = rax;
  ymm_t ymm_tmp = ymm_t(2);
  ymm_t ymm_zero = ymm_t(3);
  push(reg_ptr_global);
  mov(reg_ptr_global, reinterpret_cast<size_t>(exp_float_consts));
  vmovaps(ymm_tmp, ptr[reg_ptr_global + OFFSET_EXP_TWO]);
  vxorps(ymm_zero, ymm_zero, ymm_zero);
  vsubps(ymm_tmp, ymm_zero, ymm_tmp);
  vmulps(ymm_src, ymm_src, ymm_tmp);
  exp_ymm(ymm_src, ymm_dst);
  vmovaps(ymm_tmp, ptr[reg_ptr_global + OFFSET_EXP_ONE]);
  vaddps(ymm_dst, ymm_dst, ymm_tmp);
  vmovaps(ymm_tmp, ptr[reg_ptr_global + OFFSET_EXP_TWO]);
  vdivps(ymm_dst, ymm_tmp, ymm_dst);
  vmovaps(ymm_tmp, ptr[reg_ptr_global + OFFSET_EXP_ONE]);
  vsubps(ymm_dst, ymm_dst, ymm_tmp);
  pop(reg_ptr_global);
}

void VTanhJitCode::generate() {
  int offset = 0;
  vmovups(ymm_src, ptr[param1 + offset]);
  vtanh_ymm(ymm_src, ymm_dst);
  vmovups(ptr[param2 + offset], ymm_dst);
  ret();
}

}  // namespace gen
}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
