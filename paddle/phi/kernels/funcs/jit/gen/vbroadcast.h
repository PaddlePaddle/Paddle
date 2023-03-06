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

class VBroadcastJitCode : public JitCode {
 public:
  explicit VBroadcastJitCode(const int64_t& w,
                             size_t code_size = 256 * 1024,
                             void* code_ptr = nullptr)
      : JitCode(code_size, code_ptr), w_(w) {
    this->genCode();
  }

  DECLARE_JIT_CODE(VBroadcastJitCode);
  void genCode() override;

 private:
  int w_;
  reg64_t param_src{abi_param1};
  reg64_t param_dst{abi_param2};
  reg64_t param_h{abi_param3};
  reg64_t param_w{abi_param4};

  reg64_t reg_height{r9};
  reg64_t reg_h_i{r10};
  reg64_t reg_ptr_src_i{r11};
  reg64_t reg_ptr_dst_i{r12};
};

}  // namespace gen
}  // namespace jit
}  // namespace phi
