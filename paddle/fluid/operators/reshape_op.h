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

class ReshapeOp : public framework::OperatorWithKernel {
 public:
  ReshapeOp(const std::string &type, const framework::VariableNameMap &inputs,
            const framework::VariableNameMap &outputs,
            const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ReshapeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ReshapeOp should not be null.");

    const std::vector<int> &shape = ctx->Attrs().Get<std::vector<int>>("shape");
    PADDLE_ENFORCE(!shape.empty(),
                   "The shape information must be set by Attr(shape).");

    std::vector<int64_t> output_shape;
    auto x_dims = ctx->GetInputDim("X");
    auto out_dims = ValidateShape(shape, x_dims);
    ctx->SetOutputDim("Out", out_dims);
    // NOTE: Reshape op cannot reshape an input sequence batch into an
    // output sequence batch that has a different number of time steps. Here
    // output always shares the LoD information with input. But if
    // Attr(shape) contains 0 or -1, the actual output shape can only be
    // determined during runtime. The check for wheather it is a valid
    // output sequence batch is performed in runtime.
    ctx->ShareLoD("X", /*->*/ "Out");
  }

  static framework::DDim ValidateShape(const std::vector<int> shape,
                                       const framework::DDim &in_dims) {
    const int64_t in_size = framework::product(in_dims);
    // only one dimension canbe set to -1, whose size will be automatically
    // infered.
    const int64_t unk_dim_val = -1;
    const int64_t copy_dim_val = 0;

    std::vector<int64_t> output_shape(shape.size(), 0);
    int64_t capacity = 1;
    int unk_dim_idx = -1;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] == unk_dim_val) {
        PADDLE_ENFORCE(
            unk_dim_idx == -1,
            "Only one input dimension of Attr(shape) can be unknown.");
        unk_dim_idx = i;
      } else if (shape[i] == copy_dim_val) {
        PADDLE_ENFORCE(
            static_cast<int>(i) < in_dims.size(),
            "The index of dimension to copy from input shape must be less "
            "than the size of input shape.");
      } else {
        PADDLE_ENFORCE(
            shape[i] > 0,
            "Each input dimension of Attr(shape) must not be negtive except "
            "one unknown dimension.");
      }

      capacity *= (shape[i] ? shape[i] : in_dims[i]);
      output_shape[i] =
          (shape[i] ? static_cast<int64_t>(shape[i]) : in_dims[i]);
    }

    if (unk_dim_idx != -1) {
      output_shape[unk_dim_idx] = -in_size / capacity;
      PADDLE_ENFORCE_EQ(output_shape[unk_dim_idx] * capacity, -in_size,
                        "Invalid shape is given.");
    } else {
      PADDLE_ENFORCE_EQ(capacity, in_size, "Invalid shape is given.");
    }
    return framework::make_ddim(output_shape);
  }
};

template <typename DeviceContext, typename T>
class ReshapeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const {
    auto *out = ctx.Output<framework::LoDTensor>("Out");
    auto *in = ctx.Input<framework::LoDTensor>("X");

    auto out_dims = ReshapeOp::ValidateShape(
        ctx.Attr<std::vector<int>>("shape"), in->dims());

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
};

template <typename DeviceContext, typename T>
class ReshapeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const {
    auto *d_out = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *d_x = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

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
