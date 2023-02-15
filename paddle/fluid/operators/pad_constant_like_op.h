/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <utility>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/kernels/funcs/padding.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class PadConstantLikeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto in_x = context.Input<phi::DenseTensor>("X");
    auto in_y = context.Input<phi::DenseTensor>("Y");
    auto* out = context.Output<phi::DenseTensor>("Out");

    if (in_x->dims() == in_y->dims()) {
      framework::TensorCopy(*in_y, context.GetPlace(), out);
      return;
    }

    T pad_value = static_cast<T>(context.Attr<float>("pad_value"));
    out->mutable_data<T>(context.GetPlace());

    int rank = context.Input<phi::DenseTensor>("X")->dims().size();

    std::vector<int> pads(rank * 2, 0);

    for (int j = 0; j < rank; ++j) {
      pads[j * 2] = 0;
      pads[j * 2 + 1] = static_cast<int>(in_x->dims()[j] - in_y->dims()[j]);
    }

    phi::funcs::PaddingFunctor<DeviceContext, T>(
        rank,
        context.template device_context<DeviceContext>(),
        pads,
        pad_value,
        *in_y,
        out);
  }
};

template <typename DeviceContext, typename T>
class PadConstantLikeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto in_y = context.Input<phi::DenseTensor>("Y");
    auto in_dout =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* d_y = context.Output<phi::DenseTensor>(framework::GradVarName("Y"));

    if (d_y == nullptr) {
      return;
    }

    if (in_dout->dims() == in_y->dims()) {
      framework::TensorCopy(*in_dout, context.GetPlace(), d_y);
      return;
    }

    d_y->mutable_data<T>(context.GetPlace());
    int rank = in_dout->dims().size();

    std::vector<int> pads(static_cast<size_t>(rank) * 2, 0);
    for (int j = 0; j < rank; ++j) {
      pads[j * 2] = 0;
      pads[j * 2 + 1] = static_cast<int>(in_dout->dims()[j] - in_y->dims()[j]);
    }

    phi::funcs::PaddingGradFunctor<DeviceContext, T>(
        rank,
        context.template device_context<DeviceContext>(),
        pads,
        *in_dout,
        d_y);
  }
};

}  // namespace operators
}  // namespace paddle
