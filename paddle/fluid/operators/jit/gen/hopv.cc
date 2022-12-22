/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/jit/gen/hopv.h"

#include "paddle/fluid/operators/jit/registry.h"
#include "paddle/phi/backends/cpu/cpu_info.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {

void HOPVJitCode::genCode() {
  const int num_blocks = num_ / YMM_FLOAT_BLOCK;
  int offset = 0;

  if (num_blocks > 0) {
    // load one firstly
    vmovups(ymm_tmp, ptr[param_src]);
    offset += sizeof(float) * YMM_FLOAT_BLOCK;
    for (int i = 1; i < num_blocks; ++i) {
      vmovups(ymm_src, ptr[param_src + offset]);
      process(ymm_tmp, ymm_src, ymm_tmp);
      offset += sizeof(float) * YMM_FLOAT_BLOCK;
    }
    vextractf128(xmm_dst, ymm_tmp, 1);
    process(xmm_dst, xmm_dst, xmm_tmp);
  } else {
    if (type_ == operand_type::MAX) {
      vbroadcastss(ymm_dst, ptr[param_src]);
    } else if (type_ == operand_type::ADD) {
      vxorps(ymm_dst, ymm_dst, ymm_dst);
    }
  }

  int rest = num_ % YMM_FLOAT_BLOCK;
  if (rest >= 4) {
    vmovups(xmm_src, ptr[param_src + offset]);
    offset += sizeof(float) * 4;
    rest -= 4;
    process(xmm_dst, xmm_dst, xmm_src);
  }

  vpermilps(xmm_tmp, xmm_dst, 16 + 8 + 3);
  process(xmm_dst, xmm_dst, xmm_tmp);

  if (rest >= 2) {
    vmovq(xmm_src, ptr[param_src + offset]);
    offset += sizeof(float) * 2;
    rest -= 2;
    process(xmm_dst, xmm_dst, xmm_src);
  }

  vpermilps(xmm_tmp, xmm_dst, 1);
  process(xmm_dst, xmm_dst, xmm_tmp);

  if (rest >= 1) {
    vmovss(xmm_src, ptr[param_src + offset]);
    process(xmm_dst, xmm_dst, xmm_src);
  }
  vmovss(ptr[param_dst], xmm_dst);
  ret();
}

#define DECLARE_HOP_CREATOR(name)                                            \
  class name##Creator : public JitCodeCreator<int> {                         \
   public:                                                                   \
    bool CanBeUsed(const int& attr) const override {                         \
      return phi::backends::cpu::MayIUse(phi::backends::cpu::avx);           \
    }                                                                        \
    size_t CodeSize(const int& d) const override {                           \
      return 96 + d / YMM_FLOAT_BLOCK * 4 * 8;                               \
    }                                                                        \
    std::unique_ptr<GenBase> CreateJitCode(const int& attr) const override { \
      return make_unique<name##JitCode>(attr, CodeSize(attr));               \
    }                                                                        \
  }

DECLARE_HOP_CREATOR(HMax);
DECLARE_HOP_CREATOR(HSum);

#undef DECLARE_HOP_CREATOR

}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle

namespace gen = paddle::operators::jit::gen;

REGISTER_JITKERNEL_GEN(kHMax, gen::HMaxCreator);
REGISTER_JITKERNEL_GEN(kHSum, gen::HSumCreator);
