/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/jit/gen/vbroadcast.h"

#include "paddle/fluid/operators/jit/registry.h"
#include "paddle/phi/backends/cpu/cpu_info.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {

void VBroadcastJitCode::genCode() {
  preCode();
  constexpr int block = YMM_FLOAT_BLOCK;
  constexpr int max_num_regs = 16;
  const int num_block = w_ / block;
  const int num_groups = num_block / max_num_regs;
  const size_t block_size = sizeof(float) * block;
  std::vector<int> groups(num_groups, max_num_regs);
  int rest_num_regs = num_block % max_num_regs;
  if (rest_num_regs > 0) {
    groups.push_back(rest_num_regs);
  }

  // protect param_h
  mov(reg_height, param_h);
  Label l_next_h;
  xor_(reg_h_i, reg_h_i);
  mov(reg_ptr_dst_i, param_dst);
  L(l_next_h);
  {
    mov(reg_ptr_src_i, param_src);
    for (int num_regs : groups) {
      size_t w_offset = 0;
      for (int reg_i = 0; reg_i < num_regs; ++reg_i) {
        vmovups(ymm_t(reg_i), ptr[reg_ptr_src_i + w_offset]);
        w_offset += block_size;
      }
      add(reg_ptr_src_i, num_regs * block_size);

      w_offset = 0;
      for (int reg_i = 0; reg_i < num_regs; ++reg_i) {
        vmovups(ptr[reg_ptr_dst_i + w_offset], ymm_t(reg_i));
        w_offset += block_size;
      }
      add(reg_ptr_dst_i, num_regs * block_size);
    }  // end of groups
    inc(reg_h_i);
    cmp(reg_h_i, reg_height);
    jl(l_next_h, T_NEAR);
  }  // end of l_next_h

  postCode();
}

class VBroadcastCreator : public JitCodeCreator<int64_t> {
 public:
  bool CanBeUsed(const int64_t& w) const override {
    return phi::backends::cpu::MayIUse(phi::backends::cpu::avx) &&
           w % YMM_FLOAT_BLOCK == 0;
  }
  size_t CodeSize(const int64_t& w) const override {
    return 96 + (w / YMM_FLOAT_BLOCK) * 16 * 8;
  }
  std::unique_ptr<GenBase> CreateJitCode(const int64_t& w) const override {
    PADDLE_ENFORCE_GT(
        w,
        0,
        platform::errors::InvalidArgument(
            "The width of VBroadcast should be larger than 0. But w is %d.",
            w));
    return make_unique<VBroadcastJitCode>(w, CodeSize(w));
  }
};

}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle

namespace gen = paddle::operators::jit::gen;

REGISTER_JITKERNEL_GEN(kVBroadcast, gen::VBroadcastCreator);
