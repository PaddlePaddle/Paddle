
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

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T>
class MultiplexCPUKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto index = ins[0]->data<T>();
    auto rows = ins[1]->dims()[0];
    auto cols = ins[1]->dims()[1];
    for (auto i = 0; i < rows; i++) {
      int k = (int)index[i] + 1;
      memcpy(out->data<T>() + i * cols, ins[k]->data<T>() + i * cols,
             cols * sizeof(T));
    }
  }
};

template <typename T>
class MultiplexGradCPUKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* d_out = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto d_ins =
        ctx.MultiOutput<framework::Tensor>(framework::GradVarName("X"));
    for (size_t i = 1; i < d_ins.size(); i++) {
      if (d_ins[i]) {
        d_ins[i]->mutable_data<T>(ctx.GetPlace());
        auto dims = d_ins[i]->dims();
        memset(d_ins[i]->data<T>(), 0, framework::product(dims) * sizeof(T));
      }
    }

    auto index = ins[0]->data<T>();
    auto rows = ins[1]->dims()[0];
    auto cols = ins[1]->dims()[1];
    for (auto i = 0; i < rows; i++) {
      int k = (int)index[i] + 1;
      if (d_ins[k]) {
        memcpy(d_ins[k]->data<T>() + i * cols, d_out->data<T>() + i * cols,
               cols * sizeof(T));
      }
    }
  }
};
}
}
