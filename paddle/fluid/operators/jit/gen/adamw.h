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
#include "paddle/fluid/operators/jit/gen/jitcode.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {

class AdamWJitCode : public JitCode {
 public:
  explicit AdamWJitCode(const int& attr, size_t code_size = 256 * 1024,
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
  reg64_t reg_param_ptr{abi_param2};

  xmm_t xmm_lr = xmm_t(0);
  xmm_t xmm_lr_ratio = xmm_t(1);
  xmm_t xmm_coeff = xmm_t(2);

  ymm_t ymm_lr = ymm_t(0);
  ymm_t ymm_lr_ratio = ymm_t(1);
  ymm_t ymm_coeff = ymm_t(2);

  reg64_t reg_numel_without_tail{r10};
  reg64_t reg_offset{rax};
};

}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle
