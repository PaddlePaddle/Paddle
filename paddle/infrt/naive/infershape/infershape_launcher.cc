// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/naive/infershape/infershape_launcher.h"

namespace infrt {
namespace naive {

void InferShapedKernelLauncher::CreateKernelFrameForInferShape(
    host_context::KernelFrame* frame) {
  for (host_context::Value* value :
       frame->GetValues(0, frame->GetNumElements())) {
    // TODO(Superjomn) To extend this.
    if (value->is_type<tensor::DenseHostTensor>()) {
      values.emplace_back(MetaTensor{&value->get<tensor::DenseHostTensor>()});
      infershape_kernel_frame_builder.AddArgument(values.back().get());
    } else {
      infershape_kernel_frame_builder.AddArgument(value);
    }
  }
}

void InferShapedKernelLauncher::BuildInferShapeCache(
    const uint16_t* input_indices, const uint16_t num_inputs) {
  tensor_shape_cache.resize(num_inputs);
  for (uint16_t i = 0; i < num_inputs; i++) {
    tensor_shape_cache[i] =
        infershape_kernel_frame_builder.GetArgAt(input_indices[i])
            ->get<MetaTensor>()
            .shape();
  }
}

bool InferShapedKernelLauncher::IsShapeChanged(
    const uint16_t* input_indices, const uint16_t num_inputs) const {
  if (tensor_shape_cache.empty() && !infershape_kernel_frame_builder.IsEmpty())
    return true;

  bool changed = false;
  for (uint16_t i = 0; i < num_inputs && !changed; i++) {
    changed = changed || (tensor_shape_cache[i] !=
                          infershape_kernel_frame_builder
                              .GetArgAt<MetaTensor>(input_indices[i])
                              .shape());
  }
  return changed;
}

}  // namespace naive
}  // namespace infrt
