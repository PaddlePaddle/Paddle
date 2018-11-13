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

#include "paddle/fluid/operators/math/jit_gen.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include "paddle/fluid/platform/cpu_info.h"

DEFINE_bool(dump_jitcode, false, "Whether to dump the jitcode to file");

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {
namespace gen {

constexpr Xbyak::Operand::Code g_abi_regs[] = {
    Xbyak::Operand::RBX, Xbyak::Operand::RBP, Xbyak::Operand::R12,
    Xbyak::Operand::R13, Xbyak::Operand::R14, Xbyak::Operand::R15};

constexpr int num_g_abi_regs = sizeof(g_abi_regs) / sizeof(g_abi_regs[0]);

void JitCode::preCode() {
  for (int i = 0; i < num_g_abi_regs; ++i) {
    push(Xbyak::Reg64(g_abi_regs[i]));
  }
  if (platform::jit::MayIUse(platform::jit::avx512f)) {
    mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);
  }
}

void JitCode::postCode() {
  for (int i = 0; i < num_g_abi_regs; ++i) {
    pop(Xbyak::Reg64(g_abi_regs[num_g_abi_regs - 1 - i]));
  }
  ret();
}

void JitCode::dumpCode(const Xbyak::uint8 *code) const {
  if (code) {
    static int counter = 0;
    std::ostringstream filename;
    filename << "paddle_jitcode_" << name() << "." << counter << ".bin";
    counter++;
    std::ofstream fout(filename.str(), std::ios::out);
    if (fout.is_open()) {
      fout.write(reinterpret_cast<const char *>(code), getSize());
      fout.close();
    }
  }
}

Xbyak::Address JitCode::EVEX_compress_addr(Xbyak::Reg64 base, int offt,
                                           bool bcast) {
  int scale = 0;
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

}  // namespace gen
}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
