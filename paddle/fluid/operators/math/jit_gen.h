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

#include <gflags/gflags.h>
#include <type_traits>
#include "paddle/fluid/platform/macros.h"

#define XBYAK_USE_MMAP_ALLOCATOR
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

DECLARE_bool(dump_jitcode);

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {
namespace gen {

#define DECLARE_JIT_CODE(codename) \
  const char *name() const override { return #codename; }

// Application Binary Interface
constexpr Xbyak::Operand::Code abi_param1(Xbyak::Operand::RDI),
    abi_param2(Xbyak::Operand::RSI), abi_param3(Xbyak::Operand::RDX),
    abi_param4(Xbyak::Operand::RCX), abi_not_param1(Xbyak::Operand::RCX);

class JitCode : public Xbyak::CodeGenerator {
 public:
  explicit JitCode(size_t code_size = 256 * 1024, void *code_ptr = nullptr)
      : Xbyak::CodeGenerator(code_size, code_ptr) {}

  virtual ~JitCode() {}
  virtual const char *name() const = 0;
  virtual void generate() = 0;

  template <typename FUNC>
  const FUNC getCode() {
    this->generate();
    const Xbyak::uint8 *code = CodeGenerator::getCode();
    if (FLAGS_dump_jitcode) {
      this->dumpCode(code);
    }
    return reinterpret_cast<const FUNC>(code);
  }
  DISABLE_COPY_AND_ASSIGN(JitCode);

 protected:
  Xbyak::Reg64 param1{abi_param1};
  const int EVEX_max_8b_offt = 0x200;
  const Xbyak::Reg64 reg_EVEX_max_8b_offt = rbp;

  void preCode();
  void postCode();
  void dumpCode(const Xbyak::uint8 *code) const;
  void L(const char *label) { Xbyak::CodeGenerator::L(label); }
  void L(const Xbyak::Label &label) { Xbyak::CodeGenerator::L(label); }
  // Enhanced vector extension
  Xbyak::Address EVEX_compress_addr(Xbyak::Reg64 base, int offt,
                                    bool bcast = false);
};

}  // namespace gen
}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
