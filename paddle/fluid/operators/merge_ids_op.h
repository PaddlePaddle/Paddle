/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <tuple>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MergeIdsOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    if (!platform::is_cpu_place(place)) {
      PADDLE_THROW("MergeIds do not support GPU kernel");
    }

    auto ids = ctx.MultiInput<framework::LoDTensor>("Ids");
    auto row_ids = ctx.MultiInput<framework::LoDTensor>("Rows");
    auto x_tensors = ctx.MultiInput<framework::LoDTensor>("X");
    auto outs = ctx.MultiOutput<framework::LoDTensor>("Out");

    PADDLE_ENFORCE_EQ(row_ids.size(), x_tensors.size(),
                      "the number of Rows and X should be the same");
    PADDLE_ENFORCE_EQ(ids.size(), outs.size(),
                      "the number of Ids and Out should be the same");

    int row_ids_size = 0;
    int row_size = 0;
    int embedding_size = 0;

    for (int i = 0; i < x_tensors.size(); ++i) {
      T *x_tensor = x_tensors[i];
      T *row_id = row_ids[i];

      if (embedding_size == 0) {
        embedding_size = x_tensor->dims()[1];
      }
      PADDLE_ENFORCE_EQ(embedding_size, x_tensor->dims()[1],
                        "embedding size of all input should be the same");
      row_size += x_tensor->dims()[0];
      row_ids_size += row_id->dims()[0];
    }
    PADDLE_ENFORCE_EQ(
        row_size, row_ids_size,
        "the merged X dim[0] and merged Rows dim[0] should be the same");

    std::unordered_map<int64_t, std::tuple<T *, int64_t>> selected_rows_idx_map;
    for (int i = 0; i < x_tensors.size(); ++i) {
      T *x_tensor = x_tensors[i];
      T *row_id = row_ids[i];

      for (int j = 0; j < row_id->numel(); ++j) {
        int64_t key = row_id->data()[j];
        std::tuple<T *, int64_t> val = std::make_tuple(x_tensor, j);
        selected_rows_idx_map.insert(std::make_pair(key, val));
      }
    }
    PADDLE_ENFORCE_EQ(row_ids_size, selected_rows_idx_map.size(),
                      "the rows and tensor map size should be the same");

    for (int i = 0; i < outs.size(); ++i) {
      auto *out_ids = ids[i];
      auto *out = outs[i];
      int nums = out_ids->dims()[0];
      auto *out_data = out->mutable_data<T>(
          framework::make_ddim({nums, embedding_size}), place);
      for (int j = 0; j < nums; ++j) {
        const int id = out_data->data();
        auto *row_tuple = selected_rows_idx_map[id];
        T *row_data = row_tuple->get(0);
        int64_t row_idx = row_tuple->get(1);
        memcpy(out_data + embedding_size * j,
               row_data->data() + row_idx * embedding_size,
               sizeof(T) * embedding_size);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
