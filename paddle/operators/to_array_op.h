/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/feed_fetch_type.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class ToArrayOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    LOG(INFO) << "into compute";
    auto* input_tensor = ctx.Input<framework::Tensor>("X");
    auto input_dim = input_tensor->dims();
    auto input_dim_vec = framework::vectorize(input_dim);

    auto* table_tensor = ctx.Input<framework::Tensor>("RankSortTable");
    PADDLE_ENFORCE_EQ(true, platform::is_cpu_place(table_tensor->place()));
    auto table_dims = table_tensor->dims();
    auto* table_data = table_tensor->data<int64_t>();

    PADDLE_ENFORCE_EQ(framework::arity(table_dims), 2);
    auto table_height = table_dims[0];
    auto table_width = table_dims[1];
    PADDLE_ENFORCE_EQ(table_width, 3);

    auto max_seq_len = table_data[1] - table_data[0];
    LOG(INFO) << max_seq_len;

    auto& out = *(ctx.Output<framework::FeedFetchList>("Out"));

    LOG(INFO) << "should core";
    out.resize(max_seq_len);
    auto place = ctx.GetPlace();

    // out InferShape
    for (int64_t i = 0; i < max_seq_len; i++) {
      int64_t height = 0;
      for (int64_t j = 0; j < table_height; j++) {
        if (table_data[j * table_width + 0] + i <
            table_data[j * table_width + 1]) {
          height++;
        }
      }
      input_dim_vec[0] = height;
      out[i].Resize(framework::make_ddim(input_dim_vec));
      out[i].mutable_data<T>(place);
    }

    // out CopyFrom
    for (int64_t i = 0; i < max_seq_len; i++) {
      int out_slice_idx = 0;
      for (int64_t j = 0; j < table_height; j++) {
        int64_t input_slice_idx = table_data[j * table_width + 0] + i;
        if (input_slice_idx < table_data[j * table_width + 1]) {
          out[i]
              .Slice(out_slice_idx, out_slice_idx + 1)
              .CopyFrom(
                  input_tensor->Slice(input_slice_idx, input_slice_idx + 1),
                  place, ctx.device_context());
        }
      }
    }
  }
};

}  // namespace operators
}  // namespae paddle
