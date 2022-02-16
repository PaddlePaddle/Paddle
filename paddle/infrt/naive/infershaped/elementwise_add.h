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

#include "paddle/infrt/host_context/kernel_utils.h"
#include "paddle/infrt/naive/infershaped/infershaped_kernel_launcher.h"
#include "paddle/infrt/naive/infershaped/infershaped_utils.h"

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

static void ElementwiseAdd(tensor::DenseHostTensor* /*Context*/,
                           const tensor::DenseHostTensor& a,
                           const tensor::DenseHostTensor& b,
                           tensor::DenseHostTensor* c) {}

template <typename KernelFunc,
          KernelFunc kernel,
          typename InferShapedFunc,
          InferShapedFunc infershape>
class KernelLauncher : public InferShapedKernelLauncher {
 public:
  static const uint16_t num_input_tensors{InferShapeHelper<KernelFunc>::count};
  static const bool turn_on_infer_shape_cache{true};
  void Invoke(host_context::KernelFrame* frame) override {
    // Build the infershape KernelFrame if needed.
    // TODO(Superjomn) add unlikely here.
    if (infershape_kernel_frame_builder.IsEmpty()) {
      CreateKernelFrameForInferShape(frame);
    }
    if (turn_on_infer_shape_cache) {
      if (!turn_on_infer_shape_cache || IsShapeChanged(num_input_tensors)) {
        ::infrt::host_context::KernelImpl<InferShapedFunc, infershape>::Invoke(
            &infershape_kernel_frame_builder);
        BuildInferShapeCache(num_input_tensors);
      }
    }

    ::infrt::host_context::KernelImpl<KernelFunc, kernel>::Invoke(frame);
  }
};

}  // namespace naive
}  // namespace infrt
