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

#pragma once

#include <cmath>
#include "paddle/fluid/lite/core/context.h"
#include "paddle/fluid/lite/core/cpu_info.h"
#include "paddle/fluid/lite/core/lite_tensor.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

#ifdef __aarch64__
constexpr int MBLOCK = 8;
constexpr int NBLOCK = 12;
constexpr int KBLOCK = 4;
inline int get_hblock(ARMArch arch) { return MBLOCK; }
#else
constexpr int MBLOCK_A73 = 4;
constexpr int MBLOCK_OTH = 6;
constexpr int NBLOCK = 8;
constexpr int KBLOCK = 4;
inline int get_hblock(ARMArch arch) {
  if (arch == kA73) {
    return MBLOCK_A73;
  } else {
    return MBLOCK_OTH;
  }
}
#endif  // __aarch64__

void prepackA(float* out, const float* in, const int ldin, const int m0,
              const int mmax, const int k0, const int kmax, bool is_trans,
              ARMContext* ctx);

void prepackA(TensorLite* tout, const TensorLite& tin, int m, int k, int group,
              bool is_trans, ARMContext* ctx);

void sgemm_prepack(const float* A_packed, const float* B, const float* bias,
                   float* C, int M, int N, int K, bool is_bias, bool is_relu,
                   bool is_transB, ARMContext* ctx);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
