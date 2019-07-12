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

#include "paddle/fluid/operators/jit/gen/sgd.h"
#include <stddef.h>  // offsetof
#include <memory>
#include <vector>
#include "paddle/fluid/operators/jit/registry.h"
#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {

void SgdJitCode::genCode() {
  preCode();
  constexpr int block = YMM_FLOAT_BLOCK;
  constexpr int max_num_regs = 7;
  const int num_block = w_ / block;
  const int num_groups = num_block / max_num_regs;
  const size_t block_size = sizeof(float) * block;
  const size_t width_size = w_ * sizeof(float);
  std::vector<int> groups(num_groups, max_num_regs);
  int rest_num_regs = num_block % max_num_regs;
  if (rest_num_regs > 0) {
    groups.push_back(rest_num_regs);
  }

  vbroadcastss(ymm_lr, ptr[param_lr]);
  // protect rdx
  mov(reg_ptr_grad_i, param_grad);
  mov(reg_ptr_rows_i, param_rows);

  mov(reg_rows_size_in_byte,
      qword[param_attr + offsetof(sgd_attr_t, selected_rows_size)]);
  mov(rax, sizeof(int64_t));
  mul(reg_rows_size_in_byte);
  mov(reg_rows_size_in_byte, rax);
  add(reg_rows_size_in_byte, reg_ptr_rows_i);

  Label l_next_row;
  L(l_next_row);
  {
    mov(reg_row, qword[reg_ptr_rows_i]);
    mov(rax, width_size);
    mul(reg_row);
    mov(reg_row, rax);

    mov(reg_ptr_param_i, param_param);
    mov(reg_ptr_out_i, param_out);
    add(reg_ptr_param_i, reg_row);
    add(reg_ptr_out_i, reg_row);

    size_t w_offset = 0;
    for (int num_regs : groups) {
      // load grad
      size_t inner_offfset = w_offset;
      for (int reg_i = 0; reg_i < num_regs; ++reg_i) {
        vmovups(ymm_t(reg_i), ptr[reg_ptr_grad_i + inner_offfset]);
        inner_offfset += block_size;
      }

      // load param
      inner_offfset = w_offset;
      for (int reg_i = 0; reg_i < num_regs; ++reg_i) {
        vmovups(ymm_t(reg_i + num_regs), ptr[reg_ptr_param_i + inner_offfset]);
        inner_offfset += block_size;
      }

      // compute out
      for (int reg_i = 0; reg_i < num_regs; ++reg_i) {
        vmulps(ymm_t(reg_i), ymm_t(reg_i), ymm_lr);
        vsubps(ymm_t(reg_i + num_regs), ymm_t(reg_i + num_regs), ymm_t(reg_i));
      }

      // save out
      inner_offfset = w_offset;
      for (int reg_i = 0; reg_i < num_regs; ++reg_i) {
        vmovups(ptr[reg_ptr_out_i + inner_offfset], ymm_t(reg_i + num_regs));
        inner_offfset += block_size;
      }
      w_offset += (block_size * num_regs);
    }

    add(reg_ptr_grad_i, width_size);
    add(reg_ptr_rows_i, sizeof(int64_t));
    cmp(reg_ptr_rows_i, reg_rows_size_in_byte);
    jl(l_next_row, T_NEAR);
  }

  postCode();
}

class SgdCreator : public JitCodeCreator<sgd_attr_t> {
 public:
  bool CanBeUsed(const sgd_attr_t& attr) const override {
    return platform::MayIUse(platform::avx) &&
           attr.grad_width % YMM_FLOAT_BLOCK == 0;
  }
  size_t CodeSize(const sgd_attr_t& attr) const override {
    return 96 + (attr.grad_width / YMM_FLOAT_BLOCK) * 32 * 8;
  }
  std::unique_ptr<GenBase> CreateJitCode(
      const sgd_attr_t& attr) const override {
    PADDLE_ENFORCE_EQ(attr.param_width, attr.grad_width);
    PADDLE_ENFORCE_LE(attr.selected_rows_size, attr.grad_height);
    PADDLE_ENFORCE_GE(attr.selected_rows_size, 0);
    return make_unique<SgdJitCode>(attr, CodeSize(attr));
  }
};

}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle

namespace gen = paddle::operators::jit::gen;

REGISTER_JITKERNEL_GEN(kSgd, gen::SgdCreator);
