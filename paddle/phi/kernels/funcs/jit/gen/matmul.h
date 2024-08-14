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

#include <stdlib.h>  // for malloc and free

#include <string>
#include <vector>

#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/jit/gen/jitcode.h"

namespace phi {
namespace jit {
namespace gen {

class MatMulJitCode : public JitCode {
 public:
  explicit MatMulJitCode(const matmul_attr_t& attr,
                         size_t code_size = 256 * 1024,
                         void* code_ptr = nullptr)
      : JitCode(code_size, code_ptr), m_(attr.m), n_(attr.n), k_(attr.k) {
    PADDLE_ENFORCE_EQ(m_,
                      1,
                      common::errors::Unimplemented(
                          "Jitcode of matmul only support m==1 (first "
                          "matrix's row) now. But m is %d.",
                          m_));
    this->genCode();
  }

  std::string name() const override {
    std::string base = "MatMulJitCode";
    base = base + "_M" + std::to_string(m_) + "_N" + std::to_string(n_) + "_K" +
           std::to_string(k_);
    return base;
  }
  void genCode() override;

 private:
  int m_, n_, k_;

  reg64_t param_x{abi_param1};
  reg64_t param_y{abi_param2};
  reg64_t param_z{abi_param3};
  reg64_t param_attr{abi_param4};
  reg64_t reg_tmp{rax};

  reg64_t reg_ptr_wgt{r10};
};

}  // namespace gen
}  // namespace jit
}  // namespace phi
