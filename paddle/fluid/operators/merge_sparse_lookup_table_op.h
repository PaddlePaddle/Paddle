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

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MergeSparseLookupTableKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto inputs = ctx.MultiInput<framework::SelectedRows>("X");
    auto* out = ctx.Output<framework::SelectedRows>("Out");

    int64_t height = 0;
    int64_t ids_num = 0;
    int64_t width = 0;
    PADDLE_ENFORCE_GT(inputs.size(), 0);
    height = inputs[0]->height();
    width = inputs[0]->value().dims()[1];
    for (auto& in : inputs) {
      ids_num += in->GetIdToIndex().size();
      PADDLE_ENFORCE_EQ(in->height(), height,
                        "all input should have the same height");
    }
    T* out_data = out->mutable_value()->mutable_data<T>({height, width},
                                                        platform::CPUPlace());
    memset(out_data, 0, sizeof(T) * out->value().numel());
    out->set_height(height);

    out->InitDataShards();

    std::vector<int64_t> all_ids;
    for (auto& in : inputs) {
      auto& id_to_index = in->GetIdToIndex();
      for (auto& iter : id_to_index) {
        all_ids.push_back(iter.first);
      }
    }

    std::vector<int64_t> indexes(all_ids.size());
    out->GetIndexsByIds(all_ids, &indexes, true);

    out->SyncBeforeSave();

    for (auto& in : inputs) {
      auto& id_to_index = in->GetIdToIndex();
      const T* in_data = in->value().data<T>();
      for (auto& iter : id_to_index) {
        memcpy(out_data + out->GetIdToIndex().at(iter.first) * width,
               in_data + iter.second * width, sizeof(T) * width);
      }
    }
    out->SetVersion(1);
  }
};

}  // namespace operators
}  // namespace paddle
