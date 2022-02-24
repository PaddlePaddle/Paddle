/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_meta.h"

namespace phi {

class DenseTensorUtils {
 public:
  static DenseTensorMeta* GetMutableMeta(DenseTensor* tensor) {
    return &(tensor->meta_);
  }

  static DenseTensor Slice(const DenseTensor& tensor,
                           int64_t begin_idx,
                           int64_t end_idx) {
    size_t bytes = tensor.numel() * SizeOf(tensor.dtype());
    PADDLE_ENFORCE_GE(tensor.capacity(),
                      bytes,
                      phi::errors::InvalidArgument(
                          "The memory size %d should be enough to meet the "
                          "volume required by metadata %d.",
                          tensor.capacity(),
                          bytes));
    PADDLE_ENFORCE_GE(
        begin_idx,
        0,
        phi::errors::OutOfRange("The start row index must be greater than 0."
                                "But received the start index is d%.",
                                begin_idx));
    PADDLE_ENFORCE_LE(
        end_idx,
        tensor.dims()[0],
        phi::errors::OutOfRange("The end row index is out of bound."));
    PADDLE_ENFORCE_LT(
        begin_idx,
        end_idx,
        phi::errors::InvalidArgument(
            "The start row index must be less than the end row index."
            "But received the start index = %d, the end index = %d.",
            begin_idx,
            end_idx));
    DenseTensor ret(tensor);
    if (tensor.dims()[0] != 1) {
      ret.meta_.dims[0] = end_idx - begin_idx;
      ret.meta_.offset = tensor.meta_.offset +
                         begin_idx * (tensor.numel() / tensor.dims()[0]) *
                             paddle::experimental::SizeOf(tensor.dtype());
    }
    return ret;
  }
};

}  // namespace phi
