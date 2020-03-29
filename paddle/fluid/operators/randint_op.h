// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <random>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/uniform_random_op.h"  // only for get shape from tensor and tensorlist

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class RandintKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    std::vector<int64_t> new_shape;
    auto list_new_shape_tensor =
        ctx.MultiInput<framework::Tensor>("ShapeTensorList");
    if (list_new_shape_tensor.size() > 0 || ctx.HasInput("ShapeTensor")) {
      if (ctx.HasInput("ShapeTensor")) {
        auto* shape_tensor = ctx.Input<framework::Tensor>("ShapeTensor");
        new_shape = GetNewDataFromShapeTensor(shape_tensor);
      } else if (list_new_shape_tensor.size() > 0) {
        new_shape = GetNewDataFromShapeTensorList(list_new_shape_tensor);
      }
    }

    auto* out = ctx.Output<framework::LoDTensor>("Out");
    if (!new_shape.empty()) out->Resize(framework::make_ddim(new_shape));
    T* data = out->mutable_data<T>(ctx.GetPlace());
    int64_t size = out->numel();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(ctx.Attr<int>("low"),
                                         ctx.Attr<int>("high") - 1);
    for (int64_t i = 0; i < size; ++i) data[i] = dist(gen);
  }
};

}  // namespace operators
}  // namespace paddle
