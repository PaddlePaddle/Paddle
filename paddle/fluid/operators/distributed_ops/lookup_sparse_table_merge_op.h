/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

int64_t GetDelimiterForShard(const std::vector<int64_t>& rows, int start_idx,
                             int shard_id, int shard_num) {
  int64_t rows_num = rows.size() / 2;
  for (int64_t i = start_idx; i < rows_num; ++i) {
    if (rows[i] % shard_num != shard_id) {
      return i;
    }
  }
  return rows_num;
}

template <typename DeviceContext, typename T>
class LookupSparseTableMergeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto inputs = ctx.MultiInput<framework::SelectedRows>("X");
    auto* out = ctx.Output<framework::SelectedRows>("Out");

    int64_t height = 0;
    int64_t ids_num = 0;
    int64_t width = 0;

    height = inputs[0]->height();
    width = inputs[0]->value().dims()[1];

    for (auto& in : inputs) {
      ids_num += in->rows().size();
      height += in->height();
    }

    T* out_data = out->mutable_value()->mutable_data<T>({ids_num, width},
                                                        platform::CPUPlace());

    out->set_height(height);
    std::vector<int64_t> all_ids;
    all_ids.reserve(ids_num);
    for (auto& in : inputs) {
      all_ids.insert(all_ids.end(), in->rows().begin(), in->rows().end());
    }
    out->set_rows(all_ids);

    int64_t cnt = 0;

    for (auto& in : inputs) {
      auto rows = in->rows().size();
      const T* in_data = in->value().data<T>();
      std::copy_n(in_data, rows * width, out_data + cnt);
      cnt += rows * width;
    }
    out->SyncIndex();
  }
};

}  // namespace operators
}  // namespace paddle
