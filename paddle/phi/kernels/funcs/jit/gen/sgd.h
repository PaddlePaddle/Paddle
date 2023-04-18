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

#pragma once

#include <string>

#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/jit/gen/jitcode.h"

namespace phi {
namespace jit {
namespace gen {

class SgdJitCode : public JitCode {
 public:
  explicit SgdJitCode(const sgd_attr_t& attr,
                      size_t code_size = 256 * 1024,
                      void* code_ptr = nullptr)
      : JitCode(code_size, code_ptr), w_(attr.grad_width) {
    this->genCode();
  }

  DECLARE_JIT_CODE(SgdJitCode);
  void genCode() override;
  void mainCode(int num_regs);

 private:
  int w_;
  reg64_t param_lr{abi_param1};
  reg64_t param_param{abi_param2};
  reg64_t param_grad{abi_param3};
  reg64_t param_rows{abi_param4};
  reg64_t param_out{abi_param5};
  reg64_t param_attr{abi_param6};

  ymm_t ymm_lr = ymm_t(15);

  reg64_t reg_ptr_grad_i{r10};
  reg64_t reg_ptr_rows_i{r11};
  reg64_t reg_rows_size_in_byte{r12};
  reg64_t reg_row{r13};
  reg64_t reg_ptr_param_i{r14};
  reg64_t reg_ptr_out_i{r15};
};

}  // namespace gen
}  // namespace jit
}  // namespace phi
