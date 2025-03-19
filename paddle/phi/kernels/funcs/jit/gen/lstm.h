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

class LSTMJitCode : public VActFunc {
 public:
  explicit LSTMJitCode(bool compute_c1h1,
                       const lstm_attr_t& attr,
                       size_t code_size,
                       void* code_ptr = nullptr)
      : VActFunc(code_size, code_ptr),
        num_(attr.d),
        compute_c1h1_(compute_c1h1),
        use_peephole_(attr.use_peephole) {
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
        PADDLE_THROW(common::errors::Unimplemented(
            "Do not support jit::KernelType code: %d.", type));
      }
      return operand_type::IDENTITY;
    };
    act_gate_ = typeExchange(attr.act_gate);
    act_cand_ = typeExchange(attr.act_cand);
    act_cell_ = typeExchange(attr.act_cell);

    this->genCode();
  }

  std::string name() const override {
    std::string base = "LSTMJitCode";
    if (use_peephole_) {
      base += "_Peephole";
    }
    if (compute_c1h1_) {
      base += "_C1H1";
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
    AddTypeStr(act_cell_);
    return base;
  }
  void genCode() override;

 protected:
  int num_;
  bool compute_c1h1_;
  bool use_peephole_;
  operand_type act_gate_;
  operand_type act_cand_;
  operand_type act_cell_;
  reg64_t param1{abi_param1};
};

#define DECLARE_LSTM_JITCODE(name, compute_c1h1)                  \
  class name##JitCode : public LSTMJitCode {                      \
   public:                                                        \
    explicit name##JitCode(const lstm_attr_t& attr,               \
                           size_t code_size,                      \
                           void* code_ptr = nullptr)              \
        : LSTMJitCode(compute_c1h1, attr, code_size, code_ptr) {} \
  };

DECLARE_LSTM_JITCODE(LSTMCtHt, false);
DECLARE_LSTM_JITCODE(LSTMC1H1, true);

#undef DECLARE_LSTM_JITCODE

}  // namespace gen
}  // namespace jit
}  // namespace phi
