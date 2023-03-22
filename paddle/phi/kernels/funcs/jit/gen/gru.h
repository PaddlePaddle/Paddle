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
#include "paddle/phi/kernels/funcs/jit/gen/act.h"
#include "paddle/phi/kernels/funcs/jit/gen/jitcode.h"

namespace phi {
namespace jit {
namespace gen {

class GRUJitCode : public VActFunc {
 public:
  explicit GRUJitCode(int id,
                      const gru_attr_t& attr,
                      size_t code_size,
                      void* code_ptr = nullptr)
      : VActFunc(code_size, code_ptr), id_(id), num_(attr.d) {
    auto typeExchange = [](KernelType type) -> gen::operand_type {
      if (type == KernelType::kVSigmoid) {
        return operand_type::SIGMOID;
      } else if (type == KernelType::kVRelu) {
        return operand_type::RELU;
      } else if (type == KernelType::kVTanh) {
        return operand_type::TANH;
      } else if (type == KernelType::kVIdentity) {
        return operand_type::IDENTITY;
      } else {
        PADDLE_THROW(phi::errors::Unimplemented(
            "Do not support jit::KernelType code: %d.", type));
      }
      return operand_type::IDENTITY;
    };
    act_gate_ = typeExchange(attr.act_gate);
    act_cand_ = typeExchange(attr.act_cand);

    this->genCode();
  }

  std::string name() const override {
    std::string base = "GRUJitCode";
    if (id_ == 0) {
      base += "_H1";
    } else if (id_ == 1) {
      base += "_HtPart1";
    } else if (id_ == 2) {
      base += "_HtPart2";
    }
    auto AddTypeStr = [&](operand_type type) {
      switch (type) {
        case operand_type::RELU:
          base += "_Relu";
          break;
        case operand_type::EXP:
          base += "_Exp";
          break;
        case operand_type::SIGMOID:
          base += "_Sigmoid";
          break;
        case operand_type::TANH:
          base += "_Tanh";
          break;
        case operand_type::IDENTITY:
          base += "_Identity";
          break;
        default:
          break;
      }
    };
    AddTypeStr(act_gate_);
    AddTypeStr(act_cand_);
    return base;
  }
  void genCode() override;

 protected:
  int id_;
  int num_;
  operand_type act_gate_;
  operand_type act_cand_;
  reg64_t param1{abi_param1};
};

#define DECLARE_GRU_JITCODE(name, id)                  \
  class name##JitCode : public GRUJitCode {            \
   public:                                             \
    explicit name##JitCode(const gru_attr_t& attr,     \
                           size_t code_size,           \
                           void* code_ptr = nullptr)   \
        : GRUJitCode(id, attr, code_size, code_ptr) {} \
  };

DECLARE_GRU_JITCODE(GRUH1, 0);
DECLARE_GRU_JITCODE(GRUHtPart1, 1);
DECLARE_GRU_JITCODE(GRUHtPart2, 2);

#undef DECLARE_GRU_JITCODE

}  // namespace gen
}  // namespace jit
}  // namespace phi
