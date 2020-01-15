/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {

static inline int64_t ComputeStartIndex(int64_t start_index, int64_t size) {
  PADDLE_ENFORCE_EQ(
      start_index >= -size && start_index < size, true,
      platform::errors::InvalidArgument(
          "The start_index is expected to be in range of [%d, %d), but got %d",
          -size, size, start_index));
  if (start_index < 0) {
    start_index += size;
  }
  return start_index;
}

template <typename DeviceContext, typename T>
class PartialConcatKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    framework::Tensor* out = ctx.Output<framework::Tensor>("Out");
    PADDLE_ENFORCE_EQ(ins[0] != nullptr, true, "The input should not be null.");

    auto input_dim = ins[0]->dims();
    PADDLE_ENFORCE_EQ(input_dim.size(), 2,
                      "Only supports 2-D array with batch size in the 1st "
                      "dimension and data in the 2nd.");
    auto in_size = input_dim[1];

    // may be negative
    auto start_index = ctx.Attr<int>("start_index");
    start_index = ComputeStartIndex(start_index, in_size);

    auto partial_len = ctx.Attr<int>("length");
    if (partial_len < 0) {
      partial_len = in_size - start_index;
    }

    int batch = input_dim[0];
    int out_size = partial_len * ins.size();
    out->Resize({batch, out_size});
    auto place = ctx.GetPlace();
    T* out_data = out->mutable_data<T>(place);

    for (size_t i = 0; i < ins.size(); ++i) {
      PADDLE_ENFORCE_EQ(ins[i] && ins[i]->numel() > 0, true);
      for (int j = 0; j < batch; ++j) {
        const T* in_data = ins[i]->data<T>();
        memcpy(out_data + out_size * j + partial_len * i,
               in_data + in_size * j + start_index, partial_len * sizeof(T));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
