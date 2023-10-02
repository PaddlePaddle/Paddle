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

#pragma once

#include <string>

#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/jit/gen/jitcode.h"

namespace phi {
namespace jit {
namespace gen {

class AdamJitCode : public JitCode {
 public:
  explicit AdamJitCode(const adam_attr_t& attr,
                       size_t code_size = 256 * 1024,
                       void* code_ptr = nullptr)
      : JitCode(code_size, code_ptr) {
    this->genCode();
  }

  DECLARE_JIT_CODE(AdamJitCode);
  void genCode() override;
  void loadArgs();
  void setTailOpmask();
  void mainCode();

 private:
  reg64_t reg_numel{abi_param1};
  reg64_t reg_grad_ptr{abi_param2};
  reg64_t reg_mom1_ptr{abi_param3};
  reg64_t reg_mom2_ptr{abi_param4};
  reg64_t reg_param_ptr{abi_param5};
  reg64_t reg_mom1_out_ptr{abi_param6};

  xmm_t xmm_beta1 = xmm_t(0);
  xmm_t xmm_beta2 = xmm_t(1);
  xmm_t xmm_lr = xmm_t(2);
  xmm_t xmm_eps = xmm_t(3);
  xmm_t xmm_one_sub_beta1 = xmm_t(4);
  xmm_t xmm_one_sub_beta2 = xmm_t(5);
  xmm_t xmm_one = xmm_t(6);

  ymm_t ymm_beta1 = ymm_t(0);
  ymm_t ymm_beta2 = ymm_t(1);
  ymm_t ymm_lr = ymm_t(2);
  ymm_t ymm_eps = ymm_t(3);
  ymm_t ymm_one_sub_beta1 = ymm_t(4);
  ymm_t ymm_one_sub_beta2 = ymm_t(5);
  ymm_t ymm_one = ymm_t(6);

  reg64_t reg_mom2_out_ptr{r10};
  reg64_t reg_param_out_ptr{r11};
  reg64_t reg_numel_without_tail{r12};
  reg64_t reg_offset{rax};
};

}  // namespace gen
}  // namespace jit
}  // namespace phi
