// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>
#include <stdio.h>

extern void read_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t regs[4]);

#define TEST_BIT(reg, bit, name)     \
  do {                               \
    int set = !!(reg & (1U << bit)); \
    printf("%s %d\n", name, set);    \
  } while (0)

int main(int argc, char **argv) {
  uint32_t regs[4];
  read_cpuid(0x7, 0x0, regs);
  printf("leaf 07H: eax=0x%x ebx=0x%x ecx=0x%x, edx=0x%x\n", regs[0], regs[1],
         regs[2], regs[3]);

  TEST_BIT(regs[1], 5, "AVX2");
  TEST_BIT(regs[1], 16, "AVX512F");
  TEST_BIT(regs[1], 17, "AVX512DQ");
  TEST_BIT(regs[1], 21, "AVX512_IFMA");
  TEST_BIT(regs[1], 26, "AVX512PF (Intel® Xeon Phi™ only.)");
  TEST_BIT(regs[1], 27, "AVX512ER (Intel® Xeon Phi™ only.)");
  TEST_BIT(regs[1], 28, "AVX512CD");
  TEST_BIT(regs[1], 30, "AVX512BW");
  TEST_BIT(regs[1], 31, "AVX512VL");

  TEST_BIT(regs[2], 1, "AVX512_VBMI");
  TEST_BIT(regs[2], 6, "AVX512_VBMI2");
  TEST_BIT(regs[2], 11, "AVX512_VNNI");
  TEST_BIT(regs[2], 12, "AVX512_BITALG");
  TEST_BIT(regs[2], 14, "AVX512_VPOPCNTDQ (Intel® Xeon Phi™ only.)");

  TEST_BIT(regs[3], 2, "AVX512_4VNNIW (Intel® Xeon Phi™ only.)");
  TEST_BIT(regs[3], 3, "AVX512_4FMAPS (Intel® Xeon Phi™ only.");

  return 0;
}
