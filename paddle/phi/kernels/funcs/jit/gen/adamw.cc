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

#include "paddle/phi/kernels/funcs/jit/gen/adamw.h"

#include <cstddef>  // offsetof

#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/phi/kernels/funcs/jit/registry.h"

namespace phi::jit::gen {

void AdamWJitCode::loadArgs() {
  static constexpr int32_t one_as_float = 0x3f800000;
  static constexpr int32_t mask_all_ones = static_cast<int32_t>(0xFFFFFFFF);
  static constexpr int64_t mask_8_divisible =
      static_cast<int64_t>(0xFFFFFFFFFFFFFFF8);
  static constexpr int64_t abi_pushes_offset = num_g_abi_regs * 8;

  mov(reg_mom1_out_ptr, ptr[rsp + (abi_pushes_offset + 8)]);
  mov(reg_mom2_out_ptr, ptr[rsp + (abi_pushes_offset + 16)]);
  mov(reg_mom2_max_out_ptr, ptr[rsp + (abi_pushes_offset + 24)]);
  mov(reg_param_out_ptr, ptr[rsp + (abi_pushes_offset + 32)]);
  mov(reg_amsgrad, byte[rsp + (abi_pushes_offset + 40)]);

  mov(eax, one_as_float);
  movd(xmm_one, eax);

  vbroadcastss(ymm_one, xmm_one);                 // 1
  vbroadcastss(ymm_beta1, xmm_beta1);             // beta1
  vbroadcastss(ymm_beta2, xmm_beta2);             // beta2
  vbroadcastss(ymm_lr, xmm_lr);                   // -lr
  vbroadcastss(ymm_eps, xmm_eps);                 // eps
  vbroadcastss(ymm_old_lr, xmm_old_lr);           // old lr
  vbroadcastss(ymm_lr_ratio, xmm_lr_ratio);       // lr_ratio
  vbroadcastss(ymm_coeff, xmm_coeff);             // coeff
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

void AdamWJitCode::setTailOpmask() {
  push(r13);
  push(r14);

  mov(r13, rcx);

  mov(rcx, reg_numel);
  sub(rcx, reg_offset);  // get tail numel as float size
  shr(rcx, 2);           // as elements
  mov(r14, 1);
  shl(r14, cl);  // 2 ^ elements
  dec(r14);      // 2 ^ elements - 1, so numel first bits are set to 1
  kmovw(k1, r14d);

  mov(rcx, r13);

  pop(r14);
  pop(r13);
}

void AdamWJitCode::mainCode() {
  // load p
  vmovups(ymm10 | k1, ptr[reg_param_ptr + reg_offset]);

  // ((lr * lr_ratio) * coeff)
  vmulps(ymm11 | k1, ymm_old_lr, ymm_lr_ratio);
  vmulps(ymm11 | k1, ymm11, ymm_coeff);

  // - (lr * lr_ratio) * coeff) * p + p
  // p is stored in ymm11
  vfnmadd132ps(ymm11 | k1, ymm10, ymm10);

  // load grad
  vmovups(ymm10 | k1, ptr[reg_grad_ptr + reg_offset]);

  // beta1 * mom1 + (1 - beta1) * g
  vmulps(ymm12 | k1, ymm_one_sub_beta1, ymm10);
  vfmadd231ps(ymm12 | k1, ymm_beta1, ptr[reg_mom1_ptr + reg_offset]);

  // beta2 * mom2 + (1 - beta2) * g * g
  vmulps(ymm10 | k1, ymm10, ymm10);
  vmulps(ymm10 | k1, ymm_one_sub_beta2, ymm10);
  vfmadd231ps(ymm10 | k1, ymm_beta2, ptr[reg_mom2_ptr + reg_offset]);

  // store mom1 and mom2
  vmovups(ptr[reg_mom1_out_ptr + reg_offset] | k1, ymm12);
  vmovups(ptr[reg_mom2_out_ptr + reg_offset] | k1, ymm10);

  // // make a local label: `.without_amsgrad`
  inLocalLabel();
  // if not amsgrad then update params
  cmp(reg_amsgrad, 0);
  je(".without_amsgrad", T_NEAR);
  // load mom2_max
  vmovups(ymm13 | k1, ptr[reg_mom2_max_ptr + reg_offset]);
  // compare mom2 and mom2_max and save to mom2
  vmaxps(ymm10 | k1, ymm10, ymm13);
  // store mom2_max
  vmovups(ptr[reg_mom2_max_out_ptr + reg_offset] | k1, ymm10);

  L(".without_amsgrad");
  {
    // sqrt(mom2) + eps
    vsqrtps(ymm10 | k1, ymm10);
    vaddps(ymm10 | k1, ymm10, ymm_eps);

    // p + (-lr) * (mom1 / sqrt(mom2) + eps)
    vdivps(ymm10 | k1, ymm12, ymm10);
    vfmadd213ps(ymm10 | k1, ymm_lr, ymm11);

    // store p
    vmovups(ptr[reg_param_out_ptr + reg_offset] | k1, ymm10);
  }
  outLocalLabel();
}

void AdamWJitCode::genCode() {
  static constexpr int64_t main_loop_elems_size =
      8 * sizeof(float);  // 8 floats in YMM
  static constexpr int64_t offset_increment = main_loop_elems_size;
  preCode();
  loadArgs();

  cmp(reg_numel, main_loop_elems_size);
  jl("process_tail", T_NEAR);

  L("main_loop");
  {
    mainCode();
    add(reg_offset, offset_increment);
    cmp(reg_numel_without_tail, reg_offset);
    jg("main_loop", T_NEAR);
  }

  cmp(reg_numel, reg_offset);
  je("end", T_NEAR);  // size between jmp and label is larger than 127 byte,
                      // T_NEAR allow long jump

  L("process_tail");
  {
    setTailOpmask();
    mainCode();
  }

  L("end");
  postCode();
}

class AdamWCreator : public JitCodeCreator<adamw_attr_t> {
 public:
  bool CanBeUsed(const adamw_attr_t& attr) const override {
    return phi::backends::cpu::MayIUse(phi::backends::cpu::avx512f);
  }
  size_t CodeSize(const adamw_attr_t& attr) const override {
    return 96 + 32 * 8;
  }
  std::unique_ptr<GenBase> CreateJitCode(
      const adamw_attr_t& attr) const override {
    return make_unique<AdamWJitCode>(attr, CodeSize(attr));
  }
};

}  // namespace phi::jit::gen

namespace gen = phi::jit::gen;

REGISTER_JITKERNEL_GEN(kAdamW, gen::AdamWCreator);
