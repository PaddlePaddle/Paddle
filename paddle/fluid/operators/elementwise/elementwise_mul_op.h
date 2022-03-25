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

#include <string>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/platform/cpu_info.h"

#include "paddle/phi/kernels/elementwise_kernel.h"

namespace paddle {
namespace operators {

class ElementwiseMulOp : public ElementwiseOp {
 public:
  using Tensor = framework::Tensor;
  using ElementwiseOp::ElementwiseOp;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "X", "Y");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const {
    if (framework::IsComplexType(expected_kernel_type.data_type_)) {
      // only promote inputs’s types when contains complex input
      return framework::OpKernelType(
          framework::TransToProtoVarType(tensor.dtype()), tensor.place(),
          tensor.layout());
    } else {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), tensor.layout());
    }
  }
};

template <typename DeviceContext, typename T>
void default_elementwise_mul(const framework::ExecutionContext& ctx,
                             const framework::Tensor* x,
                             const framework::Tensor* y, framework::Tensor* z) {
  int axis = ctx.Attr<int>("axis");
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  if (x_dims.size() >= y_dims.size()) {
    ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          MulFunctor<T>(), z);
  } else {
    ElementwiseComputeEx<InverseMulFunctor<T>, DeviceContext, T>(
        ctx, x, y, axis, InverseMulFunctor<T>(), z);
  }
}

template <typename DeviceContext, typename T, class Enable = void>
struct SameDimsElemwiseMul {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor* x, const framework::Tensor* y,
                  framework::Tensor* z);
};

template <typename DeviceContext, typename T>
class ElementwiseMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x_var = ctx.InputVar("X");
    PADDLE_ENFORCE_EQ(x_var != nullptr, true,
                      platform::errors::InvalidArgument(
                          "Cannot get input Variable X, Variable name = %s.",
                          ctx.InputName("X")));
    auto* y = ctx.Input<framework::LoDTensor>("Y");

    framework::Tensor x, *z;
    if (x_var->IsType<phi::SelectedRows>()) {
      PADDLE_ENFORCE_EQ(y->dims().size() == 1 && y->dims()[0] == 1, true,
                        platform::errors::InvalidArgument(
                            "For elementwise_op, if X is Sparse, Y must be "
                            "scalar. But reveived the size of Y = %s.",
                            y->dims().size()));
      auto& x_sele = x_var->Get<phi::SelectedRows>();
      auto out_sele = ctx.Output<phi::SelectedRows>("Out");
      x = x_sele.value();
      out_sele->set_rows(x_sele.rows());
      out_sele->set_height(x_sele.height());
      out_sele->mutable_value()->Resize(x_sele.value().dims());
      out_sele->mutable_value()->mutable_data(ctx.GetPlace(), x.type());
      z = ctx.Output<phi::SelectedRows>("Out")->mutable_value();
      z->mutable_data<T>(ctx.GetPlace());
      auto dims_equal = x.dims() == y->dims();
      if (dims_equal) {
        SameDimsElemwiseMul<DeviceContext, T> same_dims_mul;
        same_dims_mul(ctx, &x, y, z);
      } else {
        default_elementwise_mul<DeviceContext, T>(ctx, &x, y, z);
      }
    } else if (x_var->IsType<framework::LoDTensor>()) {
      auto* x_lod = ctx.Input<framework::LoDTensor>("X");
      auto* z_lod = ctx.Output<framework::LoDTensor>("Out");
      z_lod->mutable_data<T>(ctx.GetPlace());

      auto& dev_ctx = ctx.device_context<DeviceContext>();
      int axis = ctx.Attr<int>("axis");
      auto pt_x = paddle::experimental::MakePhiDenseTensor(*x_lod);
      auto pt_y = paddle::experimental::MakePhiDenseTensor(*y);
      auto pt_z = paddle::experimental::MakePhiDenseTensor(*z_lod);
      phi::MultiplyRawKernel<T>(
          static_cast<const typename framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          *pt_x.get(), *pt_y.get(), axis, pt_z.get());
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "X's type[%s] is not supported by elementwise_op. X's type should be "
          "LoDTensor or SelectedRows.",
          framework::ToTypeName(x_var->Type())));
    }
  }
};

}  // namespace operators
}  // namespace paddle
