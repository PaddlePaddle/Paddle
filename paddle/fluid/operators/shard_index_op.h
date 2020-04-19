//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
template <typename T>
class ShardIndexCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    int index_num = context.Attr<int>("index_num");
    int nshards = context.Attr<int>("nshards");
    int shard_id = context.Attr<int>("shard_id");
    int ignore_value = context.Attr<int>("ignore_value");
    PADDLE_ENFORCE_GT(index_num, 0);
    PADDLE_ENFORCE_GT(nshards, 0);
    PADDLE_ENFORCE(shard_id >= 0 && shard_id < nshards,
                   "shard_id(%d) is not in range [0, %d)", shard_id, nshards);

    int shard_size = (index_num + nshards - 1) / nshards;

    out->Resize(in->dims());
    out->set_lod(in->lod());
    auto* in_data = in->data<T>();
    auto* out_data = out->mutable_data<T>(context.GetPlace());
    int64_t numel = in->numel();
    for (int64_t i = 0; i < numel; ++i) {
      PADDLE_ENFORCE(in_data[i] >= 0 && in_data[i] < index_num,
                     "Input index(%d) is out of range [0,%d)", in_data[i],
                     index_num);
      if (in_data[i] / shard_size == shard_id) {
        out_data[i] = in_data[i] % shard_size;
      } else {
        out_data[i] = ignore_value;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
