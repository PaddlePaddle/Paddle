/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/operators/jit/gen/act.h"
#include "paddle/fluid/operators/jit/gen/jitcode.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {

class LSTMJitCode : public VActJitCode {
 public:
  explicit LSTMJitCode(bool compute_c1h1, const lstm_attr_t& attr,
                       size_t code_size, void* code_ptr = nullptr)
      : VActJitCode(attr.d, operand_type::sigmoid /* this is bugy*/, code_size,
                    code_ptr),
        compute_c1h1_(compute_c1h1) {
    auto typeExchange = [](KernelType type) -> gen::operand_type {
      if (type == KernelType::vsigmoid) {
        return operand_type::sigmoid;
      } else if (type == KernelType::vrelu) {
        return operand_type::relu;
      } else if (type == KernelType::vtanh) {
        return operand_type::tanh;
      } else if (type == KernelType::videntity) {
        return operand_type::identity;
      } else {
        LOG(FATAL) << "Do not support this jit::KernelType: " << type;
      }
      return operand_type::identity;
    };
    num_ = attr.d;
    use_peephole_ = attr.use_peephole;
    act_gate_ = typeExchange(attr.act_gate);
    act_cand_ = typeExchange(attr.act_cand);
    act_cell_ = typeExchange(attr.act_cell);

    this->genCode();
  }

  const char* name() const override {
    std::string base = "LSTMJitCode";
    if (use_peephole_) {
      base += "_Peephole";
    }
    if (compute_c1h1_) {
      base += "_C1H1";
    }
    auto AddTypeStr = [&](operand_type type) {
      switch (type) {
        case operand_type::relu:
          base += "_Relu";
          break;
        case operand_type::exp:
          base += "_Exp";
          break;
        case operand_type::sigmoid:
          base += "_Sigmoid";
          break;
        case operand_type::tanh:
          base += "_Tanh";
          break;
        case operand_type::identity:
          base += "_Identity";
          break;
        default:
          break;
      }
    };
    AddTypeStr(act_gate_);
    AddTypeStr(act_cand_);
    AddTypeStr(act_cell_);
    return base.c_str();
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

#define DECLARE_LSTM_JITCODE(name, compute_c1h1)                      \
  class name##JitCode : public LSTMJitCode {                          \
   public:                                                            \
    explicit name##JitCode(const lstm_attr_t& attr, size_t code_size, \
                           void* code_ptr = nullptr)                  \
        : LSTMJitCode(compute_c1h1, attr, code_size, code_ptr) {}     \
  };

DECLARE_LSTM_JITCODE(LSTMCtHt, false);
DECLARE_LSTM_JITCODE(LSTMC1H1, true);

#undef DECLARE_LSTM_JITCODE

}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle
