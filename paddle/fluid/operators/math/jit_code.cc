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

bool VXXJitCode::init(int d, int scalar_index) {
  // It's not necessary to use avx512 since it would slow down the frequency
  // and this kernel is not compute bound.
  return MayIUse(avx) && scalar_index >= 0 && scalar_index <= 2;
}

void VXXJitCode::generate() {
  // do not need push stack, and do not need save avx512reg if do not use avx512
  int offset = 0;
  if (with_relu_) {
    vxorps(ymm_zero, ymm_zero, ymm_zero);
  }
  if (scalar_index_ == 1) {
    vbroadcastss(ymm_src1, ptr[param1]);
  } else if (scalar_index_ == 2) {
    vbroadcastss(ymm_src2, ptr[param2]);
  }
  for (int i = 0; i < num_ / AVX_FLOAT_BLOCK; ++i) {
    if (scalar_index_ != 1) {
      vmovups(ymm_src1, ptr[param1 + offset]);
    }
    if (scalar_index_ != 2) {
      vmovups(ymm_src2, ptr[param2 + offset]);
    }
    if (type_ == operand_type::mul) {
      vmulps(ymm_dst, ymm_src1, ymm_src2);
    } else if (type_ == operand_type::add) {
      vaddps(ymm_dst, ymm_src1, ymm_src2);
    }
    if (with_relu_) {
      vmaxps(ymm_dst, ymm_zero, ymm_dst);
    }
    vmovups(ptr[param3 + offset], ymm_dst);
    offset += sizeof(float) * AVX_FLOAT_BLOCK;
  }
  int rest = num_ % AVX_FLOAT_BLOCK;
  if (rest >= 4) {
    if (scalar_index_ != 1) {
      vmovups(xmm_src1, ptr[param1 + offset]);
    }
    if (scalar_index_ != 2) {
      vmovups(xmm_src2, ptr[param2 + offset]);
    }
    if (type_ == operand_type::mul) {
      vmulps(xmm_dst, xmm_src1, xmm_src2);
    } else if (type_ == operand_type::add) {
      vaddps(xmm_dst, xmm_src1, xmm_src2);
    }
    if (with_relu_) {
      vmaxps(xmm_dst, xmm_zero, xmm_dst);
    }
    vmovups(ptr[param3 + offset], xmm_dst);
    offset += sizeof(float) * 4;
    rest -= 4;
  }
  if (rest >= 2) {
    if (scalar_index_ != 1) {
      vmovups(xmm_src1, ptr[param1 + offset]);
    }
    if (scalar_index_ != 2) {
      vmovups(xmm_src2, ptr[param2 + offset]);
    }
    if (type_ == operand_type::mul) {
      vmulps(xmm_dst, xmm_src1, xmm_src2);
    } else if (type_ == operand_type::add) {
      vaddps(xmm_dst, xmm_src1, xmm_src2);
    }
    if (with_relu_) {
      vmaxps(xmm_dst, xmm_zero, xmm_dst);
    }
    vmovq(ptr[param3 + offset], xmm_dst);
    offset += sizeof(float) * 2;
    rest -= 2;
  }
  if (rest > 0) {
    if (scalar_index_ != 1) {
      vmovups(xmm_src1, ptr[param1 + offset]);
    }
    if (scalar_index_ != 2) {
      vmovups(xmm_src2, ptr[param2 + offset]);
    }
    if (type_ == operand_type::mul) {
      vmulss(xmm_dst, xmm_src1, xmm_src2);
    } else if (type_ == operand_type::add) {
      vaddss(xmm_dst, xmm_src1, xmm_src2);
    }
    if (with_relu_) {
      vmaxps(xmm_dst, xmm_zero, xmm_dst);
    }
    vmovss(ptr[param3 + offset], xmm_dst);
  }
  ret();
}

}  // namespace gen
}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
