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
#include "paddle/fluid/operators/jit/gen/jitcode.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {

// horizontal operand vector
class HOPVJitCode : public JitCode {
 public:
  explicit HOPVJitCode(int d, operand_type type, size_t code_size = 256 * 1024,
                       void* code_ptr = nullptr)
      : JitCode(code_size, code_ptr), num_(d), type_(type) {
    if (!(type_ == operand_type::MAX || type_ == operand_type::ADD)) {
      LOG(FATAL) << "Do not support this operand type: " << type_;
    }
    this->genCode();
  }

  std::string name() const override {
    std::string base = "VXXJitCode";
    if (type_ == operand_type::MAX) {
      base += "_MAX";
    } else {
      base += "_SUM";
    }
    return base;
  }
  void genCode() override;

 protected:
  template <typename JMM>
  void process(JMM& dst, JMM& src1, JMM& src2) {  // NOLINT
    if (type_ == operand_type::MAX) {
      vmaxps(dst, src1, src2);
    } else if (type_ == operand_type::ADD) {
      vaddps(dst, src1, src2);
    }
  }

 private:
  int num_;
  operand_type type_;
  reg64_t param_src{abi_param1};
  reg64_t param_dst{abi_param2};
  reg64_t param_attr{abi_param3};

  ymm_t ymm_tmp = ymm_t(0);
  ymm_t ymm_src = ymm_t(1);
  ymm_t ymm_dst = ymm_t(2);

  xmm_t xmm_tmp = xmm_t(0);
  xmm_t xmm_src = xmm_t(1);
  xmm_t xmm_dst = xmm_t(2);
};

#define DECLARE_HOP_JITCODE(name, op_type)                                    \
  class name##JitCode : public HOPVJitCode {                                  \
   public:                                                                    \
    explicit name##JitCode(int d, size_t code_size, void* code_ptr = nullptr) \
        : HOPVJitCode(d, op_type, code_size, code_ptr) {}                     \
  };

DECLARE_HOP_JITCODE(HMax, operand_type::MAX);
DECLARE_HOP_JITCODE(HSum, operand_type::ADD);

#undef DECLARE_HOP_JITCODE

}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle
