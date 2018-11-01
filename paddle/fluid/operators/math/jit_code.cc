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

#include "paddle/fluid/operators/math/jit_code.h"
#include "paddle/fluid/operators/math/jit_kernel.h"
#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {
namespace gen {

using namespace platform::jit;  // NOLINT

bool VMulJitCode::init(int d) {
  // TODO(TJ): maybe one AVX is enough, AVX above would slow down freq
  // try more with avx2 or avx512
  if (MayIUse(avx) || MayIUse(avx2)) {
    return d % AVX_FLOAT_BLOCK == 0;
  } else {
    return false;
  }
}

void VMulJitCode::generate() {
  // do not need push stack, and do not need save avx512reg if do not use avx512
  int stride = sizeof(float) * AVX_FLOAT_BLOCK;
  for (int i = 0; i < num_ / AVX_FLOAT_BLOCK; ++i) {
    vmovups(ymm_src1, ptr[param1 + i * stride]);
    vmovups(ymm_src2, ptr[param2 + i * stride]);
    vmulps(ymm_dst, ymm_src1, ymm_src2);
    vmovups(ptr[param3 + stride * i], ymm_dst);
  }
  ret();
}

}  // namespace gen
}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
