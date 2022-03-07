/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/jit/gen/adam.h"

#include <stddef.h>  // offsetof

#include "paddle/fluid/operators/jit/registry.h"
#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {

void AdamJitCode::loadArgs() {
  static constexpr int32_t one_as_float = 0x3f800000;
  static constexpr int32_t mask_all_ones = 0xFFFFFFFF;
  static constexpr int64_t mask_8_divisible = 0xFFFFFFFFFFFFFFF8;
  static constexpr int64_t abi_pushes_offset = num_g_abi_regs * 8;

  mov(reg_mom2_out_ptr, ptr[rsp + (abi_pushes_offset + 8)]);
  mov(reg_param_out_ptr, ptr[rsp + (abi_pushes_offset + 16)]);
  mov(eax, one_as_float);
  movd(xmm_one, eax);

  vbroadcastss(ymm_one, xmm_one);                 // 1
  vbroadcastss(ymm_beta1, xmm_beta1);             // beta1
  vbroadcastss(ymm_beta2, xmm_beta2);             // beta2
  vbroadcastss(ymm_lr, xmm_lr);                   // -lr
  vbroadcastss(ymm_eps, xmm_eps);                 // eps
  vsubps(ymm_one_sub_beta1, ymm_one, ymm_beta1);  // 1 - beta1
  vsubps(ymm_one_sub_beta2, ymm_one, ymm_beta2);  // 1 - beta2

  mov(reg_numel_without_tail, reg_numel);
  and_(reg_numel_without_tail, mask_8_divisible);  // make it 8-divisible

  shl(reg_numel_without_tail, 2);  // * 4 to treat it as float offset
  shl(reg_numel, 2);

  mov(eax, mask_all_ones);
  kmovw(k1, eax);

  xor_(reg_offset, reg_offset);
}

void AdamJitCode::setTailOpmask() {
  mov(r13, rcx);

  mov(rcx, reg_numel);
  sub(rcx, reg_offset);  // get tail numel as float size
  shr(rcx, 2);           // as elements
  mov(r14, 1);
  shl(r14, cl);  // 2 ^ elements
  dec(r14);      // 2 ^ elements - 1, so numel first bits are set to 1
  kmovw(k1, r14d);

  mov(rcx, r13);
}

void AdamJitCode::mainCode() {
  // load grad
  vmovups(ymm7 | k1, ptr[reg_grad_ptr + reg_offset]);

  // beta1 * mom1 + (1 - beta1) * g
  vmulps(ymm8 | k1, ymm_one_sub_beta1, ymm7);
  vfmadd231ps(ymm8 | k1, ymm_beta1, ptr[reg_mom1_ptr + reg_offset]);

  // beta2 * mom2 + (1 - beta2) * g * g
  vmulps(ymm7 | k1, ymm7, ymm7);
  vmulps(ymm7 | k1, ymm_one_sub_beta2, ymm7);
  vfmadd231ps(ymm7 | k1, ymm1, ptr[reg_mom2_ptr + reg_offset]);

  // store mom1 and mom2
  vmovups(ptr[reg_mom1_out_ptr + reg_offset] | k1, ymm8);
  vmovups(ptr[reg_mom2_out_ptr + reg_offset] | k1, ymm7);

  // sqrt(mom2) + eps
  vsqrtps(ymm7 | k1, ymm7);
  vaddps(ymm7 | k1, ymm7, ymm3);

  // p + (-lr) * (mom1 / sqrt(mom2) + eps)
  vdivps(ymm7 | k1, ymm8, ymm7);
  vfmadd213ps(ymm7 | k1, ymm2, ptr[reg_param_ptr + reg_offset]);

  // store p
  vmovups(ptr[reg_param_out_ptr + reg_offset] | k1, ymm7);
}

void AdamJitCode::genCode() {
  static constexpr int64_t main_loop_elems_size =
      8 * sizeof(float);  // 8 floats in YMM
  static constexpr int64_t offset_increment = main_loop_elems_size;
  preCode();
  loadArgs();

  cmp(reg_numel, main_loop_elems_size);
  jl("process_tail");

  L("main_loop");
  {
    mainCode();
    add(reg_offset, offset_increment);
    cmp(reg_numel_without_tail, reg_offset);
    jg("main_loop");
  }

  cmp(reg_numel, reg_offset);
  je("end");

  L("process_tail");
  {
    setTailOpmask();
    mainCode();
  }

  L("end");
  postCode();
}

class AdamCreator : public JitCodeCreator<adam_attr_t> {
 public:
  bool CanBeUsed(const adam_attr_t& attr) const override {
    return platform::MayIUse(platform::avx512f);
  }
  size_t CodeSize(const adam_attr_t& attr) const override {
    return 96 + 32 * 8;
  }
  std::unique_ptr<GenBase> CreateJitCode(
      const adam_attr_t& attr) const override {
    return make_unique<AdamJitCode>(attr, CodeSize(attr));
  }
};

}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle

namespace gen = paddle::operators::jit::gen;

REGISTER_JITKERNEL_GEN(kAdam, gen::AdamCreator);
