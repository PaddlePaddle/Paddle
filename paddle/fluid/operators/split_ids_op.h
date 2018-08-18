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

#include <unordered_map>
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
    const auto ref_vars = ctx.MultiInputVar("Refs");
    if (ids_var->IsType<framework::LoDTensor>()) {
      const auto *ids_tensor = ctx.Input<framework::LoDTensor>("Ids");
      const T *ids_data = ids_tensor->data<T>();
      auto &ids_dims = ids_tensor->dims();
      auto outs = ctx.MultiOutput<framework::LoDTensor>("Out");

      if (!ref_vars.empty()) {
        auto refs = ctx.MultiInput<framework::LoDTensor>("Refs");
        const size_t ref_num = refs.size();
        PADDLE_ENFORCE_EQ(ref_num, outs.size(),
                          "the number of Refs and Out should be the same");
        int64_t refs_ids_total_num = 0;
        int64_t input_width = ids_dims[1];
        for (size_t i = 0; i < ref_num; ++i) {
          auto *ref = refs[i];
          int64_t ref_ids_num = ref->dims()[0];
          auto *out = outs[i];
          auto *out_data = out->mutable_data<T>(
              framework::make_ddim({ref_ids_num, input_width}), place);
          memcpy(out_data, ids_data + refs_ids_total_num * input_width,
                 out->numel() * sizeof(T));
          out->set_lod(ref->lod());
          refs_ids_total_num += ref->dims()[0];
        }
      } else {
        PADDLE_ENFORCE(ids_dims[1], 1,
                       "This condition is only used for splitting input ids");
        const size_t shard_num = outs.size();
        std::vector<std::vector<T>> out_ids;
        out_ids.resize(outs.size());

        // split id by their shard_num.
        for (int i = 0; i < ids_dims[0]; ++i) {
          T id = ids_data[i];
          size_t shard_id = static_cast<size_t>(id) % shard_num;
          out_ids[shard_id].push_back(id);
        }

        // create tensor for each shard and send to parameter server
        for (size_t i = 0; i < out_ids.size(); ++i) {
          auto *shard_t = outs[i];
          std::vector<T> ids = out_ids[i];
          auto *shard_data = shard_t->mutable_data<T>(
              framework::make_ddim({static_cast<int64_t>(ids.size()), 1}),
              place);
          for (size_t i = 0; i < ids.size(); ++i) {
            shard_data[i] = ids[i];
          }
        }
      }
    } else if (ids_var->IsType<framework::SelectedRows>()) {
      PADDLE_ENFORCE(ref_vars.empty(), "");
      const auto *ids_selected_rows = ctx.Input<framework::SelectedRows>("Ids");
      auto &ids_dims = ids_selected_rows->value().dims();
      PADDLE_ENFORCE_EQ(ids_dims[0],
                        static_cast<int64_t>(ids_selected_rows->rows().size()),
                        "");
      const T *ids_data = ids_selected_rows->value().data<T>();
      const auto &ids_rows = ids_selected_rows->rows();
      auto outs = ctx.MultiOutput<framework::SelectedRows>("Out");
      const size_t shard_num = outs.size();
      for (auto &out : outs) {
        out->mutable_rows()->clear();
      }
      // get rows for outputs
      std::unordered_map<int64_t, size_t> id_to_index;
      for (size_t i = 0; i < ids_rows.size(); ++i) {
        id_to_index[ids_rows[i]] = i;
        size_t shard_id = static_cast<size_t>(ids_rows[i]) % shard_num;
        outs[shard_id]->mutable_rows()->push_back(ids_rows[i]);
      }

      int64_t row_width = ids_dims[1];
      for (auto &out : outs) {
        out->set_height(ids_selected_rows->height());
        framework::DDim ddim = framework::make_ddim(
            {static_cast<int64_t>(out->rows().size()), row_width});
        T *output = out->mutable_value()->mutable_data<T>(ddim, place);
        for (int64_t i = 0; i < ddim[0]; ++i) {
          memcpy(output + i * row_width,
                 ids_data + id_to_index[out->rows()[i]] * row_width,
                 row_width * sizeof(T));
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
