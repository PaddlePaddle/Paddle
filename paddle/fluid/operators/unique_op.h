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
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename InT>
struct UniqueOpFunctor {
  framework::Tensor* out_;
  framework::Tensor* index_;
  const framework::Tensor* in_;

  UniqueOpFunctor(framework::Tensor* out, framework::Tensor* index,
                  const framework::Tensor* in)
      : out_(out), index_(index), in_(in) {}

  template <typename IndexT>
  void apply() const {
    auto* in_data = in_->data<InT>();
    auto* index_data = index_->mutable_data<IndexT>(platform::CPUPlace());

    int64_t j = 0;

    // TODO(fangzeyang): Should optimize performance here.
    std::unordered_map<InT, int64_t> dict;
    std::vector<InT> uniq;

    PADDLE_ENFORCE(in_->numel() < pow(2, 31),
                   "numel of Unique op input should less than INT_MAX");

    for (auto i = 0; i < in_->numel(); i++) {
      auto it = dict.find(in_data[i]);
      if (it == dict.end()) {
        dict.insert(std::make_pair(in_data[i], j));
        uniq.push_back(in_data[i]);
        index_data[i] = static_cast<IndexT>(j);
        j++;
      } else {
        index_data[i] = static_cast<IndexT>(it->second);
      }
    }

    out_->Resize(framework::make_ddim({static_cast<int64_t>(uniq.size())}));
    auto out_data = out_->mutable_data<InT>(platform::CPUPlace());
    std::memcpy(out_data, uniq.data(), uniq.size() * sizeof(InT));
  }
};

template <typename T>
class UniqueKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto data_type = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("dtype"));
    auto* x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    auto* index = context.Output<framework::Tensor>("Index");

    framework::VisitDataType(data_type, UniqueOpFunctor<T>(out, index, x));
  }
};

}  // namespace operators
}  // namespace paddle
