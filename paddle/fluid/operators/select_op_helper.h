/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/platform/device_context.h"

// Functions used in SelectInputOp and SelectOutputOp
namespace paddle {
namespace operators {

// Returns the integer in mask whose numel must be 1. The integer means the
// selected branch number.
inline int GetBranchNumber(const phi::DenseTensor &mask) {
  PADDLE_ENFORCE_EQ(mask.numel(),
                    1,
                    common::errors::InvalidArgument(
                        "The numel of Input(Mask) in SelectInputOp or "
                        "SelectOutputOp must be 1. "
                        "But received %d, and it's shape is [%s].",
                        mask.numel(),
                        mask.dims()));
  if (mask.place().GetType() == phi::AllocationType::CPU) {
    return mask.data<int>()[0];
  }
  // when mask.place().GetType() == phi::AllocationType::GPU is true
  std::unique_ptr<phi::DenseTensor> cpu_mask{new phi::DenseTensor()};
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_CUSTOM_DEVICE) || defined(PADDLE_WITH_XPU)
  framework::TensorCopySync(mask, phi::CPUPlace(), cpu_mask.get());
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "This version of PaddlePaddle does NOT support GPU, "
      "but got GPU tensor 'Mask' in SelectInputOp or SelectOutputOp. "
      "Please compile PaddlePaddle WITH_GPU first."));
#endif
  return cpu_mask->data<int>()[0];
}

}  // namespace operators
}  // namespace paddle
