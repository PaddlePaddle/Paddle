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

#include "paddle/phi/kernels/funcs/jit/gen/blas.h"

#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/phi/kernels/funcs/jit/macro.h"
#include "paddle/phi/kernels/funcs/jit/registry.h"

namespace phi::jit::gen {

void VXXJitCode::genCode() {
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
    if (type_ == operand_type::MUL) {
      vmulps(ymm_dst, ymm_src1, ymm_src2);
    } else if (type_ == operand_type::ADD) {
      vaddps(ymm_dst, ymm_src1, ymm_src2);
    } else if (type_ == operand_type::SUB) {
      vsubps(ymm_dst, ymm_src1, ymm_src2);
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
      case operand_type::MUL:
        vmulps(xmm_dst, xmm_src1, xmm_src2);
        break;
      case operand_type::ADD:
        vaddps(xmm_dst, xmm_src1, xmm_src2);
        break;
      case operand_type::SUB:
        vsubps(xmm_dst, xmm_src1, xmm_src2);
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
    offset += sizeof(float) * block;  // NOLINT
    rest -= block;
  }
  ret();
}

#define DECLARE_BLAS_CREATOR(name)                                           \
  class name##Creator : public JitCodeCreator<int> {                         \
   public:                                                                   \
    bool CanBeUsed(const int& attr) const override {                         \
      return phi::backends::cpu::MayIUse(phi::backends::cpu::avx) &&         \
             attr <= 1024;                                                   \
    }                                                                        \
    size_t CodeSize(const int& d) const override {                           \
      return 96 + d / YMM_FLOAT_BLOCK * 4 * 8;                               \
    }                                                                        \
    std::unique_ptr<GenBase> CreateJitCode(const int& attr) const override { \
      return make_unique<name##JitCode>(attr, CodeSize(attr));               \
    }                                                                        \
  }

DECLARE_BLAS_CREATOR(VMul);
DECLARE_BLAS_CREATOR(VAdd);
DECLARE_BLAS_CREATOR(VSub);
DECLARE_BLAS_CREATOR(VAddRelu);
DECLARE_BLAS_CREATOR(VScal);
DECLARE_BLAS_CREATOR(VAddBias);

#undef DECLARE_BLAS_CREATOR

}  // namespace phi::jit::gen

namespace gen = phi::jit::gen;

REGISTER_JITKERNEL_GEN(kVMul, gen::VMulCreator);
REGISTER_JITKERNEL_GEN(kVAdd, gen::VAddCreator);
REGISTER_JITKERNEL_GEN(kVSub, gen::VSubCreator);
REGISTER_JITKERNEL_GEN(kVAddRelu, gen::VAddReluCreator);
REGISTER_JITKERNEL_GEN(kVScal, gen::VScalCreator);
REGISTER_JITKERNEL_GEN(kVAddBias, gen::VAddBiasCreator);
