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

#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace distributed {

inline phi::DenseTensor GetPartialTensor(const phi::DenseTensor& tensor,
                                         int64_t offset,
                                         int64_t numel) {
  phi::DenseTensor tensor_flattened;
  tensor_flattened.ShareDataWith(tensor);
  tensor_flattened.Resize({tensor.numel()});
  return tensor_flattened.Slice(offset, offset + numel);
}

void CheckTensorsShape(phi::DenseTensor* out_tensor,
                       const phi::DenseTensor& in_tensor,
                       int out_size_factor,
                       int in_size_factor) {
  // place check
  PADDLE_ENFORCE_EQ(platform::is_gpu_place(out_tensor->place()),
                    true,
                    platform::errors::InvalidArgument(
                        "Output tensor should be in GPU place."));
  PADDLE_ENFORCE_EQ(platform::is_gpu_place(in_tensor.place()),
                    true,
                    platform::errors::InvalidArgument(
                        "Input tensor should be in GPU place."));
  // shape check
  int64_t out_size = out_tensor->numel();
  PADDLE_ENFORCE_GT(out_size,
                    0,
                    platform::errors::InvalidArgument(
                        "Size of output tensor should be larger than 0."));
  int64_t in_size = in_tensor.numel();
  PADDLE_ENFORCE_GT(in_size,
                    0,
                    platform::errors::InvalidArgument(
                        "Size of input tensor should be larger than 0."));
  PADDLE_ENFORCE_EQ(
      out_size * out_size_factor,
      in_size * in_size_factor,
      platform::errors::InvalidArgument(
          "Input and output tensors should have matching sizes."));
  // dtype check
  PADDLE_ENFORCE_EQ(
      out_tensor->dtype(),
      in_tensor.dtype(),
      platform::errors::InvalidArgument(
          "Input and output tensors should have the same data type."));
}

}  //  namespace distributed
}  //  namespace paddle
