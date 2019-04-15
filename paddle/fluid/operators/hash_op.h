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
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

inline void HashOutputSize(const framework::DDim& in_dims,
                           std::vector<int64_t>& out_dims,  // NOLINT
                           int num_hash) {
  out_dims.reserve(in_dims.size() + 1);
  // copy all dims except the last one
  for (int i = 0u; i != in_dims.size() - 1; ++i) {
    out_dims.emplace_back(in_dims[i]);
  }
  out_dims.emplace_back(num_hash);
  // keep the last dim to 1
  out_dims.emplace_back(1);
}

template <typename T>
class HashKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& context) const {
    auto* out_t = context.Output<framework::LoDTensor>("Out");
    auto* in_t = context.Input<framework::LoDTensor>("X");
    int mod_by = context.Attr<int>("mod_by");
    int num_hash = context.Attr<int>("num_hash");

    auto in_dims = in_t->dims();
    auto in_lod = in_t->lod();
    PADDLE_ENFORCE_EQ(
        static_cast<uint64_t>(in_dims[0]), in_lod[0].back(),
        "The actual input data's size mismatched with LoD information.");

    std::vector<int64_t> out_dims;
    HashOutputSize(in_dims, out_dims, num_hash);
    out_t->Resize(framework::make_ddim(out_dims));
    auto* output = out_t->mutable_data<T>(context.GetPlace());

    auto seq_length = in_dims[0];
    for (int i = 1; i < in_dims.size() - 1; ++i) {
      seq_length *= in_dims[i];
    }
    auto last_dim = in_dims[in_dims.size() - 1];
    auto* input = in_t->data<T>();
    for (int idx = 0; idx < seq_length; ++idx) {
      for (int ihash = 0; ihash != num_hash; ++ihash) {
        output[idx * num_hash + ihash] =
            XXH64(input, sizeof(int) * last_dim, ihash) % mod_by;
      }
      input += last_dim;
    }
    out_t->set_lod(in_t->lod());
  }
};

}  // namespace operators
}  // namespace paddle
