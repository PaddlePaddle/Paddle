/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <vector>
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class ConcatKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    // auto* out = ctx.Output<framework::Tensor>("Out");

    const int axis = static_cast<int>(ctx.op_.GetAttr<int>("axis"));
    int N = ins.size();
    std::vector<int> offset_dim(N);
    offset_dim[0] = 0;
    for (int i = 1; i < N; i++) {
      offset_dim[i] = ins[i]->dims()[axis] + offset_dim[i - 1];
    }
    // TODO(Yancey1989): concat tensors along with specify axis
  }
};

}  // namespace operators
}  // namespace paddle
