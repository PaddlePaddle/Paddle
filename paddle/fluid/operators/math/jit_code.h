/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include "paddle/fluid/operators/math/jit_gen.h"
namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {
namespace gen {

using reg64_t = const Xbyak::Reg64;
using reg32_t = const Xbyak::Reg32;
using xmm_t = const Xbyak::Xmm;
using ymm_t = const Xbyak::Ymm;
using zmm_t = const Xbyak::Zmm;
using Label = Xbyak::Label;

// function: vec = Operand(vec, vec) (maybe with relu)
typedef enum { mul = 0, add } operand_type;

class VVVJitCode : public JitCode {
 public:
  const char* name() const override {
    std::string base = "VVVJitCode";
    if (type_ == operand_type::mul) {
      base += "_Mul";
    } else if (type_ == operand_type::add) {
      base += "_Add";
    }
    base += (with_relu_ ? "_relu" : "");
    return base.c_str();
  }
  explicit VVVJitCode(int d, operand_type type, bool with_relu,
                      size_t code_size = 256 * 1024, void* code_ptr = nullptr)
      : JitCode(code_size, code_ptr),
        num_(d),
        type_(type),
        with_relu_(with_relu) {}
  static bool init(int d);
  void generate() override;

 private:
  int num_;
  operand_type type_;
  bool with_relu_;
  reg64_t param1{abi_param1};
  reg64_t param2{abi_param2};
  reg64_t param3{abi_param3};

  xmm_t xmm_src1 = xmm_t(0);
  xmm_t xmm_src2 = xmm_t(1);
  xmm_t xmm_dst = xmm_t(1);
  xmm_t xmm_zero = xmm_t(2);

  ymm_t ymm_src1 = ymm_t(0);
  ymm_t ymm_src2 = ymm_t(1);
  ymm_t ymm_dst = ymm_t(1);
  ymm_t ymm_zero = ymm_t(2);
};

}  // namespace gen
}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
