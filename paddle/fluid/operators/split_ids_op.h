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
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    if (!platform::is_cpu_place(place)) {
      PADDLE_THROW("SplitIds do not support GPU kernel");
    }

    const auto *ids_var = ctx.InputVar("Ids");
    if (ids_var->IsType<framework::LoDTensor>()) {
      const auto &ids_dims = ctx.Input<framework::LoDTensor>("Ids")->dims();
      const T *ids = ctx.Input<framework::LoDTensor>("Ids")->data<T>();
      auto outs = ctx.MultiOutput<framework::LoDTensor>("Out");
      const size_t shard_num = outs.size();

      std::vector<std::vector<T>> out_ids;
      out_ids.resize(outs.size());

      // split id by their shard_num.
      for (int i = 0; i < ids_dims[0]; ++i) {
        T id = ids[i];
        size_t shard_id = static_cast<size_t>(id) % shard_num;
        out_ids[shard_id].push_back(id);
      }

      // create tensor for each shard and send to parameter server
      for (size_t i = 0; i < out_ids.size(); ++i) {
        auto *shard_t = outs[i];
        std::vector<T> ids = out_ids[i];
        auto *shard_data = shard_t->mutable_data<T>(
            framework::make_ddim({static_cast<int64_t>(ids.size()), 1}), place);
        for (size_t i = 0; i < ids.size(); ++i) {
          shard_data[i] = ids[i];
        }
      }
    } else if (ids_var->IsType<framework::SelectedRows>()) {
      const auto *ids_selected_rows = ctx.Input<framework::SelectedRows>("Ids");
      auto &ids_dims = ids_selected_rows->value().dims();
      PADDLE_ENFORCE_EQ(ids_dims[0],
                        static_cast<int64_t>(ids_selected_rows->rows().size()),
                        "");
      const T *ids = ids_selected_rows->value().data<T>();
      const auto &ids_rows = ids_selected_rows->rows();
      auto outs = ctx.MultiOutput<framework::SelectedRows>("Out");
      const size_t shard_num = outs.size();
      // get rows for outputs
      for (auto &id : ids_rows) {
        size_t shard_id = static_cast<size_t>(id) % shard_num;
        outs[shard_id]->mutable_rows()->push_back(id);
      }

      int64_t row_width = ids_dims[1];
      for (auto &out : outs) {
        out->set_height(ids_selected_rows->height());
        framework::DDim ddim = framework::make_ddim(
            {static_cast<int64_t>(out->rows().size()), row_width});
        T *output = out->mutable_value()->mutable_data<T>(ddim, place);
        for (int64_t i = 0; i < ddim[0]; ++i) {
          memcpy(output + i * row_width, ids + out->rows()[i] * row_width,
                 row_width * sizeof(T));
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
