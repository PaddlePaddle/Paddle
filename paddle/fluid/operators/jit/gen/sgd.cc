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

#include "paddle/fluid/operators/jit/registry.h"
#include "paddle/phi/backends/cpu/cpu_info.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {

void SgdJitCode::mainCode(int num_regs) {
  constexpr size_t block_size = sizeof(float) * YMM_FLOAT_BLOCK;
  // load grad
  for (int reg_i = 0; reg_i < num_regs; ++reg_i) {
    vmovups(ymm_t(reg_i), ptr[reg_ptr_grad_i]);
    add(reg_ptr_grad_i, block_size);
  }
  // load param
  for (int reg_i = 0; reg_i < num_regs; ++reg_i) {
    vmovups(ymm_t(reg_i + num_regs), ptr[reg_ptr_param_i]);
    add(reg_ptr_param_i, block_size);
  }
  // compute out
  for (int reg_i = 0; reg_i < num_regs; ++reg_i) {
    vmulps(ymm_t(reg_i), ymm_t(reg_i), ymm_lr);
    vsubps(ymm_t(reg_i + num_regs), ymm_t(reg_i + num_regs), ymm_t(reg_i));
  }
  // save out
  for (int reg_i = 0; reg_i < num_regs; ++reg_i) {
    vmovups(ptr[reg_ptr_out_i], ymm_t(reg_i + num_regs));
    add(reg_ptr_out_i, block_size);
  }
}

void SgdJitCode::genCode() {
  preCode();
  constexpr int block = YMM_FLOAT_BLOCK;
  constexpr int max_num_regs = 7;
  const int num_block = w_ / block;
  const int num_groups = num_block / max_num_regs;
  int rest_num_regs = num_block % max_num_regs;
  const size_t width_size = w_ * sizeof(float);

  vbroadcastss(ymm_lr, ptr[param_lr]);

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

    Label inner_loop;
    Label escape_loop;
    mov(rax, 0);
    L(inner_loop);
    {
      cmp(rax, num_groups);
      jnb(escape_loop, T_NEAR);

      mainCode(max_num_regs);

      inc(rax);
      jmp(inner_loop, T_NEAR);
    }
    L(escape_loop);
    mainCode(rest_num_regs);

    add(reg_ptr_rows_i, sizeof(int64_t));

    cmp(reg_ptr_rows_i, reg_rows_size_in_byte);
    jl(l_next_row, T_NEAR);
  }
  postCode();
}

class SgdCreator : public JitCodeCreator<sgd_attr_t> {
 public:
  bool CanBeUsed(const sgd_attr_t& attr) const override {
    return phi::backends::cpu::MayIUse(phi::backends::cpu::avx) &&
           attr.grad_width % YMM_FLOAT_BLOCK == 0;
  }
  size_t CodeSize(const sgd_attr_t& attr) const override { return 96 + 32 * 8; }
  std::unique_ptr<GenBase> CreateJitCode(
      const sgd_attr_t& attr) const override {
    PADDLE_ENFORCE_EQ(attr.param_width,
                      attr.grad_width,
                      platform::errors::InvalidArgument(
                          "The attribute param_width of Sgd should be "
                          "equal to the attribute grad_width. But param_width "
                          "is %d and grad_width is %d.",
                          attr.param_width,
                          attr.grad_width));
    PADDLE_ENFORCE_LE(attr.selected_rows_size,
                      attr.grad_height,
                      platform::errors::InvalidArgument(
                          "The attribute selected_rows_size of Sgd should be "
                          "equal to or less than the attribute grad_height. "
                          "But selected_rows_size is %d and grad_height is %d.",
                          attr.selected_rows_size,
                          attr.grad_height));
    PADDLE_ENFORCE_GE(
        attr.selected_rows_size,
        0,
        platform::errors::InvalidArgument(
            "The attribute selected_rows_size of Sgd should be "
            "equal to or larger than 0. But selected_rows_size is %d.",
            attr.selected_rows_size));
    return make_unique<SgdJitCode>(attr, CodeSize(attr));
  }
};

}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle

namespace gen = paddle::operators::jit::gen;

REGISTER_JITKERNEL_GEN(kSgd, gen::SgdCreator);
