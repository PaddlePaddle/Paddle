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
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename T>
class UniqueKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    auto* index = context.Output<framework::Tensor>("Index");

    auto* in_data = x->data<T>();
    auto* index_data = index->mutable_data<T>(platform::CPUPlace());

    int j = 0;
    std::unordered_map<T, int32_t> dict;
    std::vector<T> uniq;

    for (int i = 0; i < x->numel(); i++) {
      auto it = dict.find(in_data[i]);
      if (it == dict.end()) {
        dict.insert(std::make_pair(in_data[i], j));
        uniq.push_back(in_data[i]);
        index_data[i] = j;
        j++;
      } else {
        index_data[i] = it->second;
      }
    }

    out->Resize(framework::make_ddim({static_cast<int64_t>(uniq.size())}));
    auto out_data = out->mutable_data<T>(platform::CPUPlace());
    std::memcpy(out_data, uniq.data(), uniq.size() * sizeof(T));
  }
};

}  // namespace operators
}  // namespace paddle
