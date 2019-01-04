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

  const int group_len = max_num_regs * block * sizeof(float);
  for (int g = 0; g < num_groups; ++g) {
    pool_height<ymm_t>(g * group_len, block, max_num_regs);
  }
  if (rest_num_regs > 0) {
    pool_height<ymm_t>(num_groups * group_len, block, rest_num_regs);
  }

  // part of rest_w * height
  const int rest = w_ % block;
  pool_height_of_rest_width(rest, (w_ - rest) * sizeof(float), max_num_regs);
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
