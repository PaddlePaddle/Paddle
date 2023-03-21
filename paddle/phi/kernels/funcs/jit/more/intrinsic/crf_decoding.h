/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/jit/kernel_base.h"

namespace phi {
namespace jit {
namespace more {
namespace intrinsic {

void CRFDecoding(const int seq_len,
                 const float* x,
                 const float* w,
                 float* alpha,
                 int* track,
                 int tag_num);

class CRFDecodingKernel : public KernelMore<CRFDecodingTuple<float>> {
 public:
  CRFDecodingKernel() { this->func = CRFDecoding; }
  bool CanBeUsed(
      const typename CRFDecodingTuple<float>::attr_type&) const override;
  const char* ImplType() const override { return "Intrinsic"; }
};

}  // namespace intrinsic
}  // namespace more
}  // namespace jit
}  // namespace phi
