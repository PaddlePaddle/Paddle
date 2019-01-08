/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

extern "C" {
#include <xxhash.h>
}
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {
// template <typename DeviceContext, typename T>
template <typename T>
class HashKerel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& context) const {
    auto* out_t = context.Output<framework::LoDTensor>("Out");
    auto* in_t = context.Input<framework::LoDTensor>("X");
    int mod_by = context.Attr<int>("mod_by");
    int num_hash = context.Attr<int>("num_hash");
    auto* output = out_t->mutable_data<T>(context.GetPlace());

    auto in_dims = in_t->dims();
    auto in_lod = in_t->lod();
    PADDLE_ENFORCE_EQ(
        static_cast<uint64_t>(in_dims[0]), in_lod[0].back(),
        "The actual input data's size mismatched with LoD information.");

    auto seq_length = in_dims[0];
    auto last_dim = in_dims[in_dims.size() - 1];
    auto* input = in_t->data<T>();
    for (int idx = 0; idx < seq_length; ++idx) {
      for (int ihash = 0; ihash != num_hash; ++ihash) {
        output[idx * num_hash + ihash] =
            XXH64(input, sizeof(int) * last_dim, ihash) % mod_by;
      }
      input += last_dim;
    }
  }
};

}  // namespace operators
}  // namespace paddle
