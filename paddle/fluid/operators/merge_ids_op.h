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
    VLOG(3) << "run in MergeIdsOpKernel";

    const auto *ids_var = ctx.InputVar("Ids");
    PADDLE_ENFORCE(ids_var->IsType<framework::LoDTensor>(),
                   "only support to merge Ids of LoDTensor");

    const auto &ids_tensor = ids_var->Get<framework::LoDTensor>();
    const auto &ids_dims = ids_tensor.dims();
    const int64_t *ids = ids_tensor.data<int64_t>();

    auto x_tensors = ctx.MultiInput<framework::LoDTensor>("X");

    auto *out = ctx.Output<framework::LoDTensor>("Out");

    int batch_size = 0;
    int embedding_size = 0;
    for (auto &input : x_tensors) {
      if (framework::product(input->dims()) != 0) {
        if (embedding_size == 0) {
          embedding_size = input->dims()[1];
        }
        PADDLE_ENFORCE_EQ(embedding_size, input->dims()[1],
                          "embedding size of all input should be the same");
        batch_size += input->dims()[0];
      }
    }
    PADDLE_ENFORCE_EQ(
        batch_size, ids_dims[0],
        "the batch size of ids and merged embedding value should be the same");

    const size_t shard_num = x_tensors.size();

    if (shard_num == 1) {
      VLOG(3) << "only one shard, we can copy the data directly";
      TensorCopy(*x_tensors[0], place, out);
    } else {
      std::vector<int> in_indexs(shard_num, 0);
      auto *out_data = out->mutable_data<T>(
          framework::make_ddim({batch_size, embedding_size}), place);
      // copy data from ins[shard_num] to out.
      for (int i = 0; i < ids_dims[0]; ++i) {
        int64_t id = ids[i];
        size_t shard_id = static_cast<size_t>(id) % shard_num;
        int index = in_indexs[shard_id];
        memcpy(out_data + embedding_size * i,
               x_tensors[shard_id]->data<T>() + index * embedding_size,
               sizeof(T) * embedding_size);
        in_indexs[shard_id] += 1;
      }

      for (size_t i = 0; i < shard_num; ++i) {
        PADDLE_ENFORCE_EQ(in_indexs[i], x_tensors[i]->dims()[0],
                          "after merge, all data in x_tensor should be used");
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
