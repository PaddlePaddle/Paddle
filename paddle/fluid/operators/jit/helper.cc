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

#include "paddle/fluid/operators/jit/helper.h"
#include <algorithm>  // tolower
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace jit {

#define ONE_CASE(key) \
  case key:           \
    return #key

const char* to_string(KernelType kt) {
  switch (kt) {
    ONE_CASE(kNone);
    ONE_CASE(kVMul);
    ONE_CASE(kVAdd);
    ONE_CASE(kVAddRelu);
    ONE_CASE(kVSub);
    ONE_CASE(kVScal);
    ONE_CASE(kVAddBias);
    ONE_CASE(kVRelu);
    ONE_CASE(kVIdentity);
    ONE_CASE(kVExp);
    ONE_CASE(kVSquare);
    ONE_CASE(kVSigmoid);
    ONE_CASE(kVTanh);
    ONE_CASE(kLSTMCtHt);
    ONE_CASE(kLSTMC1H1);
    ONE_CASE(kGRUH1);
    ONE_CASE(kGRUHtPart1);
    ONE_CASE(kGRUHtPart2);
    ONE_CASE(kCRFDecoding);
    ONE_CASE(kLayerNorm);
    ONE_CASE(kNCHW16CMulNC);
    ONE_CASE(kSeqPool);
    ONE_CASE(kMatMul);
    default:
      PADDLE_THROW("Not support type: %d, or forget to add it.", kt);
      return "NOT JITKernel";
  }
  return nullptr;
}

const char* to_string(SeqPoolType tp) {
  switch (tp) {
    ONE_CASE(kNonePoolType);
    ONE_CASE(kSum);
    ONE_CASE(kAvg);
    ONE_CASE(kSqrt);
    default:
      PADDLE_THROW("Not support type: %d, or forget to add it.", tp);
      return "NOT PoolType";
  }
  return nullptr;
}
#undef ONE_CASE

KernelType to_kerneltype(const std::string& act) {
  std::string lower = act;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  if (lower == "relu" || lower == "vrelu") {
    return kVRelu;
  } else if (lower == "identity" || lower == "videntity" || lower == "") {
    return kVIdentity;
  } else if (lower == "exp" || lower == "vexp") {
    return kVExp;
  } else if (lower == "sigmoid" || lower == "vsigmoid") {
    return kVSigmoid;
  } else if (lower == "tanh" || lower == "vtanh") {
    return kVTanh;
  }
  PADDLE_THROW("Not support type: %s, or forget to add this case", act);
  return kNone;
}

}  // namespace jit
}  // namespace operators
}  // namespace paddle
