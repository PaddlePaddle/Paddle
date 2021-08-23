/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/diag_v2_op.h"
#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

class DiagV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "diag_v2");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "diag_v2");

    auto x_dims = ctx->GetInputDim("X");
    auto offset = ctx->Attrs().Get<int>("offset");

    if (x_dims.size() == 1UL) {
      int64_t size_ = x_dims[0] + std::abs(offset);
      ctx->SetOutputDim("Out", {size_, size_});
    } else if (x_dims.size() == 2UL) {
      int64_t size_ = 0;
      if (offset >= 0) {
        // Note(LutaoChu): Do not use std::min here, otherwise the calculation
        // of `size_` will have unexpected result on Windows Python3.8
        if (x_dims[0] < x_dims[1] - offset) {
          size_ = x_dims[0];
        } else {
          size_ = x_dims[1] - offset;
        }
      } else {
        // Note(LutaoChu): Do not use std::min here, otherwise the calculation
        // of `size_` will have unexpected result on Windows Python3.8
        if (x_dims[0] + offset < x_dims[1]) {
          size_ = x_dims[0] + offset;
        } else {
          size_ = x_dims[1];
        }
      }
      ctx->SetOutputDim("Out", {size_});
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The input tensor X's dimensions of DiagV2Op should be either 1 or "
          "2, but received %d.",
          x_dims.size()));
    }
  }
};

class DiagV2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor. Its shape is either 1-D or 2-D.");
    AddOutput("Out", "The output tensor. A square matrix or a vector.");
    AddAttr<int>("offset",
                 "The diagonal offset. A positive value represents "
                 "superdiagonal, 0 represents the main diagonal, and a "
                 "negative value represents subdiagonal.")
        .SetDefault(0);
    AddAttr<float>("padding_value",
                   "Use this value to fill the area outside the specified "
                   "diagonal band. Only takes effect when the input is a 1-D "
                   "Tensor. The default value is 0.")
        .SetDefault(0.0f);
    AddComment(R"DOC(
      If ``x`` is a vector (1-D tensor), a 2-D square tensor with the elements of ``x`` as the diagonal is returned.

      If ``x`` is a matrix (2-D tensor), a 1-D tensor with the diagonal elements of ``x`` is returned.

      The argument ``offset`` controls the diagonal offset:

      If ``offset`` = 0, it is the main diagonal.

      If ``offset`` > 0, it is superdiagonal.

      If ``offset`` < 0, it is subdiagonal.
)DOC");
  }
};

template <typename DeviceContext, typename T>
class DiagV2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<framework::Tensor>("X");
    auto* x_data = X->data<T>();
    auto x_dims = X->dims();
    int offset = context.Attr<int>("offset");
    auto* out = context.Output<framework::Tensor>("Out");
    T* out_data = out->mutable_data<T>(context.GetPlace());
    auto out_dims = out->dims();

    int64_t i;
    if (x_dims.size() == 1) {
      float padding_value = context.Attr<float>("padding_value");
      math::SetConstant<DeviceContext, T> set_padding_value;
      auto& dev_ctx = context.template device_context<DeviceContext>();
      set_padding_value(dev_ctx, out, static_cast<T>(padding_value));

      auto x_length = x_dims[0];
      const int& x_stride = ComputeStride(0, x_dims);

      auto out_stride_0 = ComputeStride(0, out_dims);
      auto out_stride_1 = ComputeStride(1, out_dims);
      out_data +=
          (offset >= 0 ? offset * out_stride_1 : -offset * out_stride_0);

      for (i = 0; i < x_length; i++) {
        out_data[i * (out_stride_0 + out_stride_1)] = x_data[i * x_stride];
      }
    } else {
      auto out_length = out_dims[0];
      const int& x_stride_0 = ComputeStride(0, x_dims);
      const int& x_stride_1 = ComputeStride(1, x_dims);

      auto out_stride_0 = ComputeStride(0, out_dims);
      x_data += (offset >= 0 ? offset * x_stride_1 : -offset * x_stride_0);
      for (i = 0; i < out_length; i++) {
        out_data[i * out_stride_0] = x_data[i * (x_stride_0 + x_stride_1)];
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    diag_v2, ops::DiagV2Op, ops::DiagV2OpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    diag_v2, ops::DiagV2Kernel<paddle::platform::CPUDeviceContext, int>,
    ops::DiagV2Kernel<paddle::platform::CPUDeviceContext, float>,
    ops::DiagV2Kernel<paddle::platform::CPUDeviceContext, double>,
    ops::DiagV2Kernel<paddle::platform::CPUDeviceContext, int64_t>);
