/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/cpu/cpu_info.h"

#ifdef PADDLE_WITH_XBYAK
#include "xbyak/xbyak_util.h"
#endif

namespace phi {
namespace backends {
namespace cpu {

size_t CpuMinChunkSize() {
  // Allow to allocate the minimum chunk size is 4 KB.
  return 1 << 12;
}

#ifdef PADDLE_WITH_XBYAK
static Xbyak::util::Cpu cpu;
bool MayIUse(const cpu_isa_t cpu_isa) {
  using namespace Xbyak::util;  // NOLINT
  switch (cpu_isa) {
    case sse42:
      return cpu.has(Cpu::tSSE42);
    case avx:
      return cpu.has(Cpu::tAVX);
    case avx2:
      return cpu.has(Cpu::tAVX2);
    case avx512f:
      return cpu.has(Cpu::tAVX512F);
    case avx512_core:
      return true && cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) &&
             cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ);
    case avx512_core_vnni:
      return true && cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) &&
             cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ) &&
             cpu.has(Cpu::tAVX512_VNNI);
    case avx512_mic:
      return true && cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512CD) &&
             cpu.has(Cpu::tAVX512ER) && cpu.has(Cpu::tAVX512PF);
    case avx512_mic_4ops:
      return true && MayIUse(avx512_mic) && cpu.has(Cpu::tAVX512_4FMAPS) &&
             cpu.has(Cpu::tAVX512_4VNNIW);
    case avx512_bf16:
      return true && cpu.has(Cpu::tAVX512_BF16);
    case isa_any:
      return true;
  }
  return false;
}
#else
bool MayIUse(const cpu_isa_t cpu_isa) {
  if (cpu_isa == isa_any) {
    return true;
  } else {
#if !defined(WITH_NV_JETSON) && !defined(PADDLE_WITH_ARM) && \
    !defined(PADDLE_WITH_SW) && !defined(PADDLE_WITH_MIPS)
    int reg[4];
    cpuid(reg, 0);
    int nIds = reg[0];
    if (nIds >= 0x00000001) {
      // EAX = 1
      cpuid(reg, 0x00000001);
      // AVX: ECX Bit 28
      if (cpu_isa == avx) {
        int avx_mask = (1 << 28);
        return (reg[2] & avx_mask) != 0;
      }
    }
    if (nIds >= 0x00000007) {
      // EAX = 7
      cpuid(reg, 0x00000007);
      if (cpu_isa == avx2) {
        // AVX2: EBX Bit 5
        int avx2_mask = (1 << 5);
        return (reg[1] & avx2_mask) != 0;
      } else if (cpu_isa == avx512f) {
        // AVX512F: EBX Bit 16
        int avx512f_mask = (1 << 16);
        return (reg[1] & avx512f_mask) != 0;
      } else if (cpu_isa == avx512_core) {
        unsigned int avx512f_mask = (1 << 16);
        unsigned int avx512dq_mask = (1 << 17);
        unsigned int avx512bw_mask = (1 << 30);
        unsigned int avx512vl_mask = (1 << 31);
        return ((reg[1] & avx512f_mask) && (reg[1] & avx512dq_mask) &&
                (reg[1] & avx512bw_mask) && (reg[1] & avx512vl_mask));
      }
      // EAX = 7, ECX = 1
      cpuid(reg, 0x00010007);
      if (cpu_isa == avx512_bf16) {
        // AVX512BF16: EAX Bit 5
        int avx512bf16_mask = (1 << 5);
        return (reg[0] & avx512bf16_mask) != 0;
      }
    }
#endif
    return false;
  }
}
#endif

}  // namespace cpu
}  // namespace backends
}  // namespace phi
