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
#include <type_traits>
#include "paddle/fluid/operators/jit/gen_base.h"
#include "paddle/fluid/platform/cpu_info.h"

#define XBYAK_USE_MMAP_ALLOCATOR
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace paddle {
namespace operators {
namespace jit {
namespace gen {

// Application Binary Interface
constexpr Xbyak::Operand::Code abi_param1(Xbyak::Operand::RDI),
    abi_param2(Xbyak::Operand::RSI), abi_param3(Xbyak::Operand::RDX),
    abi_param4(Xbyak::Operand::RCX), abi_param5(Xbyak::Operand::R8),
    abi_param6(Xbyak::Operand::R9);

constexpr Xbyak::Operand::Code g_abi_regs[] = {
    Xbyak::Operand::RBX, Xbyak::Operand::RBP, Xbyak::Operand::R12,
    Xbyak::Operand::R13, Xbyak::Operand::R14, Xbyak::Operand::R15};

constexpr int num_g_abi_regs = sizeof(g_abi_regs) / sizeof(g_abi_regs[0]);

using reg64_t = const Xbyak::Reg64;
using reg32_t = const Xbyak::Reg32;
using xmm_t = const Xbyak::Xmm;
using ymm_t = const Xbyak::Ymm;
using zmm_t = const Xbyak::Zmm;
using opmask_t = const Xbyak::Opmask;
using Label = Xbyak::Label;

typedef enum {
  MUL = 0,
  MAX,
  ADD,
  SUB,
  RELU,
  EXP,
  SQUARE,
  SIGMOID,
  TANH,
  IDENTITY
} operand_type;

#define DECLARE_JIT_CODE(codename) \
  std::string name() const override { return #codename; }

class JitCode : public GenBase, public Xbyak::CodeGenerator {
 public:
  explicit JitCode(size_t code_size, void* code_ptr = nullptr)
      : Xbyak::CodeGenerator(
            (code_size % 4096 != 0 ? (code_size / 4096 + 1) * 4096 : code_size),
            code_ptr) {}

  virtual void genCode() = 0;

  size_t getSize() const override { return CodeGenerator::getSize(); }
  const unsigned char* getCodeInternal() const override {
    const Xbyak::uint8* code = CodeGenerator::getCode();
    return code;
  }

 protected:
  Xbyak::Reg64 param1{abi_param1};
  const int EVEX_max_8b_offt = 0x200;
  const Xbyak::Reg64 reg_EVEX_max_8b_offt = rbp;

  virtual void preCode() {
    for (int i = 0; i < num_g_abi_regs; ++i) {
      push(Xbyak::Reg64(g_abi_regs[i]));
    }
    if (platform::MayIUse(platform::avx512f)) {
      mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);
    }
  }
  virtual void postCode() {
    for (int i = 0; i < num_g_abi_regs; ++i) {
      pop(Xbyak::Reg64(g_abi_regs[num_g_abi_regs - 1 - i]));
    }
    ret();
  }
  void L(const char* label) { Xbyak::CodeGenerator::L(label); }
  void L(Xbyak::Label& label) { Xbyak::CodeGenerator::L(label); }  // NOLINT
  // Enhanced vector extension
  Xbyak::Address EVEX_compress_addr(Xbyak::Reg64 base, int offt,
                                    bool bcast = false) {
    int scale = 0;
    // Learn from https://github.com/intel/mkl-dnn
    if (EVEX_max_8b_offt <= offt && offt < 3 * EVEX_max_8b_offt) {
      offt = offt - 2 * EVEX_max_8b_offt;
      scale = 1;
    } else if (3 * EVEX_max_8b_offt <= offt && offt < 5 * EVEX_max_8b_offt) {
      offt = offt - 4 * EVEX_max_8b_offt;
      scale = 2;
    }
    auto re = Xbyak::RegExp() + base + offt;
    if (scale) {
      re = re + reg_EVEX_max_8b_offt * scale;
    }
    if (bcast) {
      return zword_b[re];
    } else {
      return zword[re];
    }
  }
};

}  // namespace gen
}  // namespace jit
}  // namespace operators
}  // namespace paddle
