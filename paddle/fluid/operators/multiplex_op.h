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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MultiplexCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto ids = ctx.Input<framework::Tensor>("Ids");
    auto* out = ctx.Output<framework::Tensor>("Out");

    out->mutable_data<T>(ctx.GetPlace());

    for (size_t i = 0; i < ins.size(); ++i) {
      PADDLE_ENFORCE_GT(
          ins[i]->numel(), 0,
          platform::errors::OutOfRange(
              "indexing will be out of bounds with size 0 for the %d-th input.",
              i));
    }

    auto rows = ins[0]->dims()[0];
    auto cols = ins[0]->numel() / rows;
    auto index = ids->data<int32_t>();
    platform::CPUPlace place = ctx.GetPlace();
    for (auto i = 0; i < rows; i++) {
      int32_t k = index[i];
      PADDLE_ENFORCE_GE(k, 0, platform::errors::PreconditionNotMet(
                                  "index must be nonnegative."));
      PADDLE_ENFORCE_LT(static_cast<size_t>(k), ins.size(),
                        platform::errors::PreconditionNotMet(
                            "index exceeds the number of candidate tensors."));
      memory::Copy(place, out->data<T>() + i * cols, place,
                   ins[k]->data<T>() + i * cols, cols * sizeof(T));
    }
  }
};

template <typename DeviceContext, typename T>
class MultiplexGradCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* d_out = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* ids = ctx.Input<framework::Tensor>("Ids");
    auto d_ins =
        ctx.MultiOutput<framework::Tensor>(framework::GradVarName("X"));

    size_t idx = -1UL;
    for (size_t i = 0; i < d_ins.size(); i++) {
      if (d_ins[i]) {
        d_ins[i]->mutable_data<T>(ctx.GetPlace());
        auto t = framework::EigenVector<T>::Flatten(*d_ins[i]);
        t.device(*ctx.template device_context<DeviceContext>().eigen_device()) =
            t.constant(static_cast<T>(0));

        idx = i;
      }
    }

    if (idx == -1UL) return;

    auto rows = d_ins[idx]->dims()[0];
    auto cols = d_ins[idx]->numel() / rows;
    auto* index = ids->data<int32_t>();
    platform::CPUPlace place = ctx.GetPlace();
    for (auto i = 0; i < rows; i++) {
      size_t k = static_cast<size_t>(index[i]);
      if (d_ins[k]) {
        memory::Copy(place, d_ins[k]->data<T>() + i * cols, place,
                     d_out->data<T>() + i * cols, cols * sizeof(T));
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle
