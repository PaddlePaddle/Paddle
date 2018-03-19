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

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ReshapeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    auto* in = ctx.Input<framework::LoDTensor>("X");

    auto out_dims =
        ValidateShape(ctx.Attr<std::vector<int>>("shape"), in->dims());

    if (!in->lod().empty()) {
      PADDLE_ENFORCE_EQ(
          out_dims[0], in->dims()[0],
          "Reshape operator cannot reshape an input sequence batch "
          "into an output sequence batch that has a different "
          "number of time steps. Please consider using "
          "sequence_reshape op.");
    }

    bool inplace = ctx.Attr<bool>("inplace");
    if (!inplace) {
      out->mutable_data<T>(ctx.GetPlace());
      framework::TensorCopy(*in, ctx.GetPlace(), ctx.device_context(), out);
      out->Resize(out_dims);
    } else {
      out->ShareDataWith(*in);
      out->Resize(out_dims);
    }
  }

 private:
  framework::DDim ValidateShape(const std::vector<int> shape_attr,
                                const framework::DDim& in_dims) const {
    const int64_t in_size = framework::product(in_dims);
    // only one dimension canbe set to -1, whose size will be automatically
    // infered.
    const int64_t unknown_index = -1;

    std::vector<int64_t> output_shape(shape_attr.size(), 0);
    int64_t capacity = 1;
    int neg_dim_idx = -1;
    for (size_t i = 0; i < shape_attr.size(); ++i) {
      if (shape_attr[i] == unknown_index) neg_dim_idx = i;
      capacity *= (shape_attr[i] ? shape_attr[i] : in_dims[i]);
      output_shape[i] =
          (shape_attr[i] ? static_cast<int64_t>(shape_attr[i]) : in_dims[i]);
    }

    if (neg_dim_idx != -1) {
      output_shape[neg_dim_idx] = -in_size / capacity;
      PADDLE_ENFORCE_EQ(output_shape[neg_dim_idx] * capacity, -in_size,
                        "Invalid shape is given.");
    } else {
      PADDLE_ENFORCE_EQ(capacity, in_size, "Invalid shape is given.");
    }
    return framework::make_ddim(output_shape);
  }
};

template <typename DeviceContext, typename T>
class ReshapeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* d_out = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* d_x = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    d_x->mutable_data<T>(ctx.GetPlace());
    bool inplace = ctx.Attr<bool>("inplace");

    auto in_dims = d_x->dims();
    if (!inplace) {
      framework::TensorCopy(*d_out, ctx.GetPlace(), ctx.device_context(), d_x);
      d_x->Resize(in_dims);
    } else {
      d_x->ShareDataWith(*d_out);
      d_x->Resize(in_dims);
    }
  }
};
}  // namespace operators
}  // namespace paddle
