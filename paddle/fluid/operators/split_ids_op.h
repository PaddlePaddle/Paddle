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
#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SplitIdsOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto place = ctx.GetPlace();
    if (!platform::is_cpu_place(place)) {
      PADDLE_THROW("SplitIds do not support GPU kernel");
    }

    const auto* ids_t = ctx.Input<framework::LoDTensor>("Ids");
    auto& ids_dims = ids_t->dims();
    auto outs = ctx.MultiOutput<framework::LoDTensor>("Out");

    const T* ids = ids_t->data<T>();

    const size_t shard_num = outs.size();

    std::vector<std::vector<T>> out_ids;
    out_ids.resize(outs.size());

    // split id by their shard_num.
    for (size_t i = 0; i < ids_dims[0]; ++i) {
      T id = ids[i];
      size_t shard_id = static_cast<size_t>(id) % shard_num;
      out_ids[shard_id].push_back(id);
    }

    // create tensor for each shard and send to parameter server
    for (size_t i = 0; i < out_ids.size(); ++i) {
      auto* shard_t = outs[i];
      std::vector<T> ids = out_ids[i];
      auto* shard_data = shard_t->mutable_data<T>(
          framework::make_ddim({static_cast<int64_t>(ids.size()), 1}), place);
      for (size_t i = 0; i < ids.size(); ++i) {
        shard_data[i] = ids[i];
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
