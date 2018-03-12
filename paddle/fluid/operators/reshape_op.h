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
    auto* out = ctx.Output<framework::Tensor>("Out");
    auto* in = ctx.Input<framework::Tensor>("X");

    auto* shape = ctx.Input<framework::Tensor>("Shape");
    framework::DDim out_dims;
    if (shape) {
      std::vector<int64_t> output_shape;
      ValidateShape(*shape, framework::product(in->dims()), output_shape);

      out_dims = framework::make_ddim(output_shape);
    } else {
      out_dims = out->dims();
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
  void ValidateShape(const framework::Tensor& shape, const int64_t in_size,
                     std::vector<int64_t>& output_shape) const {
    std::vector<size_t> neg_dims_idx;
    const int unknown_index = -1;  // only one dimension canbe set to -1, whose
                                   // size will be automatically infered.

    const int64_t dimension = shape.dims()[1];
    std::cout << "dimension =" << dimension << std::endl;
    const T* shape_data = shape.data<T>();

    for (int64_t i = 0; i < dimension; ++i) {
      PADDLE_ENFORCE(shape_data[i] > 1 || shape_data[i] == unknown_index,
                     "Each input dimension of Attr(shape) must be positive, or "
                     "only one input dimension can be -1.");
      if (shape_data[i] == unknown_index) neg_dims_idx.push_back(i);
    }
    PADDLE_ENFORCE_LE(
        neg_dims_idx.size(), 1,
        "Only one input dimension of Attr(shape) can be unknown.");

    int64_t capacity = 1;
    output_shape.resize(dimension, 0);
    for (int64_t i = 0; i < dimension; ++i) {
      capacity *= shape_data[i];
      output_shape[i] = static_cast<int64_t>(shape_data[i]);
    }

    if (neg_dims_idx.size())
      output_shape[neg_dims_idx[0]] = in_size / (-capacity);
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
