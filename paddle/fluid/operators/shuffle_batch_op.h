// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <atomic>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/phi/core/mixed_vector.h"

namespace paddle {
namespace operators {

template <typename T>
using Vector = phi::Vector<T>;

template <typename T, typename DeviceContext>
class ShuffleBatchKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {}
};

template <typename T, typename DeviceContext>
class ShuffleBatchGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *out_grad =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto *shuffleidx = context.Input<phi::DenseTensor>("ShuffleIdx");
    auto *x_grad =
        context.Output<phi::DenseTensor>(framework::GradVarName("X"));

    auto embed_size = out_grad->dims()[out_grad->dims().size() - 1];
    auto elem_size = 1;
    for (auto i = 0; i < out_grad->dims().size() - 1; i++)
      elem_size *= out_grad->dims()[i];

    std::vector<int> idx_vec_grad(elem_size);
    auto *shuffleidx_data = shuffleidx->data<int64_t>();
    for (size_t i = 0; i < idx_vec_grad.size(); i++) {
      idx_vec_grad[shuffleidx_data[i]] = i;
    }

    // copy data according to idx_vec_grad
    auto *out_grad_data = out_grad->data<T>();
    auto *x_grad_data = x_grad->mutable_data<T>(context.GetPlace());
    for (auto i = 0; i < elem_size; i++) {
      memcpy(x_grad_data + idx_vec_grad[i] * embed_size,
             out_grad_data + i * embed_size,
             embed_size * sizeof(T));
    }
  }
};
}  // namespace operators
}  // namespace paddle
