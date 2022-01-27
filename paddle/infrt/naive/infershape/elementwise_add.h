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
#pragma once

#include <llvm/ADT/SmallVector.h>

#include "paddle/infrt/naive/infershape/infershape_launcher.h"

// This file contains a example of the infershape ElementwiseAdd kernel.
// Some of the following code should be generated from PTEN by script.

namespace infrt {
namespace naive {

static void ElementwiseAddInferShape(const MetaTensor& a,
                                     const MetaTensor& b,
                                     MetaTensor* c) {
  CHECK(a.shape() == b.shape())
      << "ElementwiseAdd, but shapes of a b are not match";
  *c->mutable_shape() = a.shape();
}

// TODO(zhiqiang) This class should be generated.
class ElementwiseAddInferShapeLauncher : public InferShapeLauncher {
 public:
  constexpr static uint16_t input_tensor_indice[2] = {0, 1};
  static const uint16_t num_input_tensors = 2;
  static const bool turn_on_infer_shape_cache{true};

  void Invoke(host_context::KernelFrame* frame) override {
    // Build the infershape KernelFrame if needed.
    // TODO(Superjomn) add unlikely here.
    if (infershape_kernel_frame_builder.IsEmpty()) {
      CreateKernelFrameForInferShape(frame);
    }
    if (turn_on_infer_shape_cache) {
      if (IsShapeChanged(input_tensor_indice, num_input_tensors)) {
        ElementwiseAddInferShape(
            infershape_kernel_frame_builder.GetArgAt<MetaTensor>(0),
            infershape_kernel_frame_builder.GetArgAt<MetaTensor>(1),
            &infershape_kernel_frame_builder.GetArgAt(2)->get<MetaTensor>());
        BuildInferShapeCache(input_tensor_indice, num_input_tensors);
      }
    } else {
      BuildInferShapeCache(input_tensor_indice, num_input_tensors);
    }
  }
};

}  // namespace naive
}  // namespace infrt
