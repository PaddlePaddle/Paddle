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

#include <type_traits>
#include "paddle/fluid/operators/jitkernels/kernels.h"

#define XBYAK_USE_MMAP_ALLOCATOR
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace paddle {
namespace operators {
namespace jitkernels {
namespace jitcode {

// Application Binary Interface
constexpr Xbyak::Operand::Code abi_param1(Xbyak::Operand::RDI),
    abi_param2(Xbyak::Operand::RSI), abi_param3(Xbyak::Operand::RDX),
    abi_param4(Xbyak::Operand::RCX), abi_not_param1(Xbyak::Operand::RCX);

template <KernelType KT, typename Attr>
class JitCode : public JitBase, public Xbyak::CodeGenerator {
 public:
  JitCode(Attr attr, size_t code_size, void* code_ptr = nullptr)
      : Xbyak::CodeGenerator(code_size, code_ptr) {
    this->genCode();
  }

  virtual const char* name() const = 0;
  virtual void genCode() = 0;

  const unsigned char* getCodeInternal() override {
    const Xbyak::uint8* code = CodeGenerator::getCode();
    return code;
  }
};

}  // namespace jitcode
}  // namespace jitkernels
}  // namespace operators
}  // namespace paddle
