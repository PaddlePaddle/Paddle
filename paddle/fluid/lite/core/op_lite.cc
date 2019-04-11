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

#include "paddle/fluid/lite/core/op_lite.h"
#include "op_lite.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {

std::vector<std::unique_ptr<KernelBase>> OpLite::CreateKernels(
    const std::vector<OpLite::Place> &places, const std::string &kernel_type) {
  std::vector<std::unique_ptr<KernelBase>> kernels;
  CHECK(!op_type_.empty()) << "op_type_ should be set first";

  for (auto place : places) {
    kernels.emplace_back(KernelRegistry::Global().Create(
        (kernel_type.empty() ? op_type_ : kernel_type), place.target,
        place.precision));
  }

  return kernels;
}

void OpLite::PickKernel(const std::vector<OpLite::Place> &valid_places,
                        OpLite::KernelStrategy kernel_strategy) {
  switch (kernel_strategy) {
    case KernelStrategy::kStatic:
      StaticPickKernel(valid_places);
      break;
    default:
      LOG(FATAL) << "unsupported kernel strategy";
  }
}

}  // namespace lite
}  // namespace paddle
