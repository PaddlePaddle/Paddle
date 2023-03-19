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

// function: vec = Operand(vec(or scalar), vec(or scalar)) (maybe with relu)
class VXXJitCode : public JitCode {
 public:
  explicit VXXJitCode(int d,
                      operand_type type,
                      int scalar_index,
                      bool with_relu,
                      size_t code_size = 256 * 1024,
                      void* code_ptr = nullptr)
      : JitCode(code_size, code_ptr),
        num_(d),
        type_(type),
        scalar_index_(scalar_index),
        with_relu_(with_relu) {
    if (!(type_ == operand_type::MUL || type_ == operand_type::ADD ||
          type_ == operand_type::SUB)) {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Do not support operand type code: %d.", type));
    }
    this->genCode();
  }

  std::string name() const override {
    std::string base = "VXXJitCode";
    if (scalar_index_ == 1) {
      base += "_Scalar";
    } else {
      base += "_Vec";
    }
    if (type_ == operand_type::MUL) {
      base += "_Mul";
    } else if (type_ == operand_type::ADD) {
      base += "_Add";
    } else if (type_ == operand_type::SUB) {
      base += "_SUB";
    }
    if (scalar_index_ == 2) {
      base += "_Scalar";
    } else {
      base += "_Vec";
    }
    base += (with_relu_ ? "_Relu" : "");
    base += "_D" + std::to_string(num_);
    return base;
  }
  void genCode() override;

 private:
  int num_;
  operand_type type_;
  int scalar_index_;
  bool with_relu_;
  reg64_t param1{abi_param1};
  reg64_t param2{abi_param2};
  reg64_t param3{abi_param3};

  xmm_t xmm_src1 = xmm_t(0);
  xmm_t xmm_src2 = xmm_t(1);
  xmm_t xmm_dst = xmm_t(2);
  xmm_t xmm_zero = xmm_t(3);

  ymm_t ymm_src1 = ymm_t(0);
  ymm_t ymm_src2 = ymm_t(1);
  ymm_t ymm_dst = ymm_t(2);
  ymm_t ymm_zero = ymm_t(3);
};

#define DECLARE_BLAS_JITCODE(name, op_type, scalar_idx, with_relu)             \
  class name##JitCode : public VXXJitCode {                                    \
   public:                                                                     \
    explicit name##JitCode(int d, size_t code_size, void* code_ptr = nullptr)  \
        : VXXJitCode(d, op_type, scalar_idx, with_relu, code_size, code_ptr) { \
    }                                                                          \
  };

DECLARE_BLAS_JITCODE(VMul, operand_type::MUL, 0, false);
DECLARE_BLAS_JITCODE(VAdd, operand_type::ADD, 0, false);
DECLARE_BLAS_JITCODE(VSub, operand_type::SUB, 0, false);
DECLARE_BLAS_JITCODE(VAddRelu, operand_type::ADD, 0, true);
DECLARE_BLAS_JITCODE(VScal, operand_type::MUL, 1, false);
DECLARE_BLAS_JITCODE(VAddBias, operand_type::ADD, 1, false);

#undef DECLARE_BLAS_JITCODE

}  // namespace gen
}  // namespace jit
}  // namespace phi
