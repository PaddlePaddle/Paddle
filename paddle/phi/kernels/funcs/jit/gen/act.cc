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

#include "paddle/phi/kernels/funcs/jit/gen/act.h"
#include <array>

#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/phi/kernels/funcs/jit/registry.h"

namespace phi::jit::gen {

const float ALIGN32_BEG exp_float_consts[] ALIGN32_END = {  // NOLINT
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

const int ALIGN32_BEG exp_int_0x7f[] ALIGN32_END = {  // NOLINT
    REPEAT_8TIMES(0x7f)};                             // NOLINT
int ALIGN32_BEG g_tmp_mem[16] ALIGN32_END = {0};      // NOLINT

void VActJitCode::genCode() {
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
    offset += sizeof(float) * block;  // NOLINT
    rest -= block;
  }
  ret();
}

#define DECLARE_ACT_CREATOR(name)                                            \
  class name##Creator : public JitCodeCreator<int> {                         \
   public:                                                                   \
    bool CanBeUsed(const int& attr) const override;                          \
    size_t CodeSize(const int& d) const override;                            \
    std::unique_ptr<GenBase> CreateJitCode(const int& attr) const override { \
      return make_unique<name##JitCode>(attr, CodeSize(attr));               \
    }                                                                        \
  }

DECLARE_ACT_CREATOR(VRelu);
DECLARE_ACT_CREATOR(VSquare);
DECLARE_ACT_CREATOR(VIdentity);
DECLARE_ACT_CREATOR(VExp);
DECLARE_ACT_CREATOR(VSigmoid);
DECLARE_ACT_CREATOR(VTanh);

// TODO(TJ): tuning use me
bool VReluCreator::CanBeUsed(const int& d) const {
  return phi::backends::cpu::MayIUse(phi::backends::cpu::avx);
}

bool VSquareCreator::CanBeUsed(const int& d) const {
  return phi::backends::cpu::MayIUse(phi::backends::cpu::avx);
}

bool VIdentityCreator::CanBeUsed(const int& d) const {
  return phi::backends::cpu::MayIUse(phi::backends::cpu::avx);
}

bool VExpCreator::CanBeUsed(const int& d) const {
  return phi::backends::cpu::MayIUse(phi::backends::cpu::avx) && d < 32;
}

bool VSigmoidCreator::CanBeUsed(const int& d) const {
  return phi::backends::cpu::MayIUse(phi::backends::cpu::avx);
}

bool VTanhCreator::CanBeUsed(const int& d) const {
  return phi::backends::cpu::MayIUse(phi::backends::cpu::avx);
}

size_t VReluCreator::CodeSize(const int& d) const {
  return 96 /* init size */ + (d / YMM_FLOAT_BLOCK + 3) * 4 /* instructions */ *
                                  8 /* average bytes for each instruction */;
}

size_t VSquareCreator::CodeSize(const int& d) const {
  return 96 + (d / YMM_FLOAT_BLOCK + 3) * 4 * 8;
}

size_t VIdentityCreator::CodeSize(const int& d) const {
  return 96 + (d / YMM_FLOAT_BLOCK + 3) * 4 * 8;
}

size_t VExpCreator::CodeSize(const int& d) const {
  return 96 + (d / YMM_FLOAT_BLOCK + 3) * 70 * 8;
}

size_t VSigmoidCreator::CodeSize(const int& d) const {
  return 96 + (d / YMM_FLOAT_BLOCK + 3) * 82 * 8;
}

size_t VTanhCreator::CodeSize(const int& d) const {
  return 96 + (d / YMM_FLOAT_BLOCK + 3) * 84 * 8;
}

#undef DECLARE_ACT_CREATOR

}  // namespace phi::jit::gen

namespace gen = phi::jit::gen;

REGISTER_JITKERNEL_GEN(kVRelu, gen::VReluCreator);
REGISTER_JITKERNEL_GEN(kVSquare, gen::VSquareCreator);
REGISTER_JITKERNEL_GEN(kVIdentity, gen::VIdentityCreator);
REGISTER_JITKERNEL_GEN(kVExp, gen::VExpCreator);
REGISTER_JITKERNEL_GEN(kVSigmoid, gen::VSigmoidCreator);
REGISTER_JITKERNEL_GEN(kVTanh, gen::VTanhCreator);
