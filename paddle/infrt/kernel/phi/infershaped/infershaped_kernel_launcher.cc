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

#include "paddle/infrt/kernel/phi/infershaped/infershaped_kernel_launcher.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/meta_tensor.h"

namespace infrt {
namespace kernel {

void InferShapedKernelLauncher::CreateKernelFrameForInferShape(
    host_context::KernelFrame* frame) {
  for (host_context::Value* value :
       frame->GetValues(1, frame->GetNumElements() - 1)) {
    // TODO(Superjomn) To extend this.
    if (value->is_type<::phi::DenseTensor>()) {
      values.emplace_back(new host_context::Value{
          ::phi::MetaTensor{&value->get<::phi::DenseTensor>()}});
      infershape_kernel_frame_builder.AddArgument(values.back().get());
    } else {
      infershape_kernel_frame_builder.AddArgument(value);
    }
  }
  if (infershape_kernel_frame_builder.GetNumArgs() < arg_size_) {
    infershape_kernel_frame_builder.AddArgument(
        new host_context::Value(::phi::MetaConfig()));
  }
}

void InferShapedKernelLauncher::BuildInferShapeCache(
    const uint16_t num_inputs) {
  tensor_shape_cache.resize(num_inputs);
  for (uint16_t i = 0; i < num_inputs; i++) {
    tensor_shape_cache[i] = infershape_kernel_frame_builder.GetArgAt(i)
                                ->get<::phi::MetaTensor>()
                                .dims();
  }
}

bool InferShapedKernelLauncher::IsShapeChanged(
    const uint16_t num_inputs) const {
  if (tensor_shape_cache.empty() && !infershape_kernel_frame_builder.IsEmpty())
    return true;

  bool changed = false;
  for (uint16_t i = 0; i < num_inputs && !changed; i++) {
    changed =
        changed ||
        (tensor_shape_cache[i] !=
         infershape_kernel_frame_builder.GetArgAt<::phi::MetaTensor>(i).dims());
  }
  return changed;
}

}  // namespace kernel
}  // namespace infrt
