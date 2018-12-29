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

#include "paddle/fluid/operators/jit/gen/seqpool.h"
#include "paddle/fluid/operators/jit/registry.h"
#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {

void SeqPoolJitCode::genCode() {
  constexpr int block = YMM_FLOAT_BLOCK;
  constexpr int max_num_regs = 8;
  const int num_block = w_ / block;
  const int num_groups = num_block / max_num_regs;
  int rest_num_regs = num_block % max_num_regs;
  if (type_ == SeqPoolType::kAvg) {
    float scalar = 1.f / h_;
    mov(reg32_scalar, scalar);
  } else if (type_ == SeqPoolType::kSqrt) {
    float scalar = 1.f / std::sqrt(static_cast<float>(h_));
    mov(reg32_scalar, scalar);
  }

  // TODO(TJ): make height load from params
  const int group_len = max_num_regs * block * sizeof(float);
  for (int g = 0; g < num_groups; ++g) {
    pool_height<ymm_t>(g * group_len, block, max_num_regs);
  }
  if (rest_num_regs > 0) {
    pool_height<ymm_t>(num_groups * group_len, block, rest_num_regs);
  }

  // rest part
  const int rest = w_ % block;
  const bool has_block4 = rest / 4 > 0;
  const bool has_block2 = (rest % 4) / 2 > 0;
  const bool has_block1 = (rest % 2) == 1;
  const int w_offset = num_block * YMM_FLOAT_BLOCK * sizeof(float);
  for (int h = 0; h < h_; ++h) {
    int offset = h * w_ * sizeof(float) + w_offset;
    const int shift_regs = (h == 0) ? 0 : max_num_regs;
    int reg_idx = 0;
    if (has_block4) {
      vmovups(xmm_t(reg_idx + shift_regs), ptr[param1 + offset]);
      offset += sizeof(float) * 4;
      reg_idx++;
    }
    if (has_block2) {
      vmovq(xmm_t(reg_idx + shift_regs), ptr[param1 + offset]);
      offset += sizeof(float) * 2;
      reg_idx++;
    }
    if (has_block1) {
      vmovss(xmm_t(reg_idx + shift_regs), ptr[param1 + offset]);
      reg_idx++;
    }
    rest_num_regs = reg_idx;
    if (h > 0) {
      for (int i = 0; i < reg_idx; ++i) {
        vaddps(xmm_t(i), xmm_t(i), xmm_t(i + max_num_regs));
      }
    }
  }
  // save right now
  int offset = w_offset;
  if (type_ == SeqPoolType::kAvg || type_ == SeqPoolType::kSqrt) {
    vbroadcastss(xmm_t(max_num_regs - 1), reg32_scalar);
    for (int i = 0; i < rest_num_regs; ++i) {
      vmulps(xmm_t(i), xmm_t(i), xmm_t(max_num_regs - 1));
    }
  }
  int reg_idx = 0;
  if (has_block4) {
    vmovups(ptr[param2 + offset], xmm_t(reg_idx));
    offset += sizeof(float) * 4;
    reg_idx++;
  }
  if (has_block2) {
    vmovq(ptr[param2 + offset], xmm_t(reg_idx));
    offset += sizeof(float) * 2;
    reg_idx++;
  }
  if (has_block1) {
    vmovss(ptr[param2 + offset], xmm_t(reg_idx));
  }
  ret();
}

class SeqPoolCreator : public JitCodeCreator<seq_pool_attr_t> {
 public:
  bool UseMe(const seq_pool_attr_t& attr) const override {
    return platform::MayIUse(platform::avx);
  }
  size_t CodeSize(const seq_pool_attr_t& attr) const override {
    // TODO(TJ): remove attr.h when enabled height
    bool yes =
        attr.type == SeqPoolType::kAvg || attr.type == SeqPoolType::kSqrt;
    return 96 /* basic */ +
           ((attr.w / YMM_FLOAT_BLOCK + 4 /* rest */) * 2 /* for sum */
            * (attr.h + (yes ? 3 : 1 /*for avg or sqrt*/))) *
               8;
  }
  std::unique_ptr<GenBase> CreateJitCode(
      const seq_pool_attr_t& attr) const override {
    PADDLE_ENFORCE_GT(attr.w, 0);
    PADDLE_ENFORCE_GT(attr.h, 0);
    return make_unique<SeqPoolJitCode>(attr, CodeSize(attr));
  }
};

}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle

namespace gen = paddle::operators::jit::gen;

REGISTER_JITKERNEL_GEN(kSeqPool, gen::SeqPoolCreator);
