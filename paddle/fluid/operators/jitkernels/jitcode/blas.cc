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
#include "paddle/fluid/operators/jitkernels/jitcode/blas.h"
#include "paddle/fluid/operators/jitkernels/registry.h"

namespace paddle {
namespace operators {
namespace jitkernels {
namespace jitcode {

void VXXJitCode::genCode() {
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
  for (int i = 0; i < num_ / YMM_FLOAT_BLOCK; ++i) {
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
    offset += sizeof(float) * YMM_FLOAT_BLOCK;
  }
  int rest = num_ % YMM_FLOAT_BLOCK;
  while (rest > 0) {
    int block = XMM_FLOAT_BLOCK;
    if (rest >= 4) {
      block = 4;
      if (scalar_index_ != 1) {
        vmovups(xmm_src1, ptr[param1 + offset]);
      }
      if (scalar_index_ != 2) {
        vmovups(xmm_src2, ptr[param2 + offset]);
      }
    } else if (rest >= 2) {
      block = 2;
      if (scalar_index_ != 1) {
        vmovq(xmm_src1, ptr[param1 + offset]);
      }
      if (scalar_index_ != 2) {
        vmovq(xmm_src2, ptr[param2 + offset]);
      }
    } else {
      block = 1;
      if (scalar_index_ != 1) {
        vmovss(xmm_src1, ptr[param1 + offset]);
      }
      if (scalar_index_ != 2) {
        vmovss(xmm_src2, ptr[param2 + offset]);
      }
    }
    switch (type_) {
      case operand_type::mul:
        vmulps(xmm_dst, xmm_src1, xmm_src2);
        break;
      case operand_type::add:
        vaddps(xmm_dst, xmm_src1, xmm_src2);
        break;
      default:
        break;
    }
    if (with_relu_) {
      vmaxps(xmm_dst, xmm_zero, xmm_dst);
    }
    if (rest >= 4) {
      vmovups(ptr[param3 + offset], xmm_dst);
    } else if (rest >= 2) {
      vmovq(ptr[param3 + offset], xmm_dst);
    } else {
      vmovss(ptr[param3 + offset], xmm_dst);
    }
    offset += sizeof(float) * block;
    rest -= block;
  }
  ret();
}

}  // namespace jitcode

template <>
std::unique_ptr<JitBase> CreateJitCode<KernelType::vmul, float, int>(int attr) {
  if (UseJitCode<KernelType::vmul, float, int>(attr)) {
    return make_unique<jitcode::VMulJitCode>(
        attr, CodeSize<KernelType::vmul, float, int>(attr));
  }
  return nullptr;
}

}  // namespace jitkernels
}  // namespace operators
}  // namespace paddle
