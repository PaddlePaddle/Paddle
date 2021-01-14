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
    PADDLE_ENFORCE_GT(
        index_num, 0,
        platform::errors::InvalidArgument(
            "The value 'index_num' for Op(shard_index) must be greater than 0, "
            "but the value given is %d.",
            index_num));
    PADDLE_ENFORCE_GT(nshards, 0,
                      platform::errors::InvalidArgument(
                          "The value 'nshard' for Op(shard_index) must be "
                          "greater than 0, but the value given is %d.",
                          nshards));
    PADDLE_ENFORCE_GE(
        shard_id, 0,
        platform::errors::InvalidArgument(
            "The value 'shard_id' for Op(shard_index) must be greater or "
            "equal to 0, but the value given is %d.",
            shard_id));
    PADDLE_ENFORCE_LT(
        shard_id, nshards,
        platform::errors::InvalidArgument(
            "The value 'shard_id' for Op(shard_index) must be less than "
            "nshards (%d), but the value given is %d.",
            nshards, shard_id));

    int shard_size = (index_num + nshards - 1) / nshards;

    out->Resize(in->dims());
    out->set_lod(in->lod());
    auto* in_data = in->data<T>();
    auto* out_data = out->mutable_data<T>(context.GetPlace());
    int64_t numel = in->numel();
    for (int64_t i = 0; i < numel; ++i) {
      PADDLE_ENFORCE_GE(in_data[i], 0,
                        platform::errors::InvalidArgument(
                            "The input_index for Op(shard_index) must be "
                            "greater or equal to 0, but the value given is %d.",
                            in_data[i]));
      PADDLE_ENFORCE_LT(in_data[i], index_num,
                        platform::errors::InvalidArgument(
                            "The input_index for Op(shard_index) must be less "
                            "than index_num (%d), but the value given is %d.",
                            index_num, in_data[i]));
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
