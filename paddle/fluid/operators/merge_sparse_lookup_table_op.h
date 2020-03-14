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

int64_t GetDelimiterForShard(const std::vector<int64_t>& rows, int start_idx, int shard_id, int shard_num) {
  int64_t rows_num = rows.size() / 2;
  for(int64_t i = start_idx; i < rows_num; ++i) {
    if (rows[i] % shard_num != shard_id) {
      return i;
    }
  }
  return rows_num;
}

template <typename DeviceContext, typename T>
class MergeSparseLookupTableKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto inputs = ctx.MultiInput<framework::SelectedRows>("X");
    auto* out = ctx.Output<framework::SelectedRows>("Out");

    int64_t height = 0;
    int64_t rows_num = 0;
    int64_t width = 0;
    PADDLE_ENFORCE_GT(inputs.size(), 0);

    width = inputs[0]->value().dims()[1];

    for (auto& in : inputs) {
      rows_num += in->rows().size();
      height += in->height();
    }

    T* out_data = out->mutable_value()->mutable_data<T>({height, width},
                                                        platform::CPUPlace());

    out->set_height(height);
    std::vector<int64_t> all_rows;
    all_rows.reserve(rows_num);

    int64_t cnt = 0;
    std::vector<int64_t> start_indexs;
    start_indexs.reserve(inputs.size());
    for(int j = 0; j < inputs.size(); ++j) {
      PADDLE_ENFORCE_EQ(inputs[j]->rows().size() % 2, 0, "rows should have n * 2 elements");
      start_indexs[j] = 0;
    }

    auto shard_num = out->shard_num();
    int64_t out_shard_size = height / shard_num;

    std::vector<int64_t> indexs;
    indexs.reserve(rows_num/2);
    for(int i = 0; i < shard_num; ++i) {
      for(int j = 0; j < inputs.size(); ++j) {
        auto start_index = start_indexs[j];
        auto end_index = GetDelimiterForShard(inputs[j]->rows(), start_index, i, shard_num);
        int64_t in_shard_size = inputs[j]->height() / shard_num; 
        auto ids_num = inputs[j]->rows().size() / 2;
        const T* in_data = inputs[j]->value().data<T>(); 
        std::copy_n(in_data + i * in_shard_size * width, (end_index - start_index) * width, out_data + (i * out_shard_size + cnt) * width);
        for(int m = start_index; m < end_index; ++m) {
          auto original_index = inputs[j]->rows()[ids_num + m];
          auto original_offset = original_index - i * in_shard_size;
          auto idx = i * out_shard_size + cnt + original_offset; 
          PADDLE_ENFORCE_LT(idx, height,
                        "idx should be less then table height");
          all_rows.emplace_back(inputs[j]->rows()[m]);
          indexs.emplace_back(idx);
        }
        cnt += end_index - start_index;
        start_indexs[j] = end_index;
      }
      cnt = 0;
    } 
    all_rows.insert(all_rows.end(), indexs.begin(), indexs.end());
    out->set_rows(all_rows);
  }
};

}  // namespace operators
}  // namespace paddle
