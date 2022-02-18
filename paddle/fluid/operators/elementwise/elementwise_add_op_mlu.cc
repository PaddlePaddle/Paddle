/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename T>
class ElementwiseAddMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    int axis = ctx.Attr<int>("axis");
    const auto& x_dims = x->dims();
    const auto& y_dims = y->dims();
    axis = (axis < 0 ? (std::abs(x_dims.size() - y_dims.size()) + axis + 1)
                     : axis);
    int max_dim = std::max(x_dims.size(), y_dims.size());
    std::vector<int> x_dims_array(max_dim);
    std::vector<int> y_dims_array(max_dim);
    std::vector<int> out_dims_array(max_dim);
    GetBroadcastDimsArrays(x_dims, y_dims, x_dims_array.data(),
                           y_dims_array.data(), out_dims_array.data(), max_dim,
                           axis);

    MLUCnnlTensorDesc x_desc(max_dim, x_dims_array.data(),
                             ToCnnlDataType(x->type()));
    MLUCnnlTensorDesc y_desc(max_dim, y_dims_array.data(),
                             ToCnnlDataType(y->type()));
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnlOpTensorDesc op_tensor_desc(CNNL_OP_TENSOR_ADD, ToCnnlDataType<T>(),
                                       CNNL_NOT_PROPAGATE_NAN);

    MLUCnnl::OpTensor(ctx, op_tensor_desc.get(), x_desc.get(), GetBasePtr(x),
                      y_desc.get(), GetBasePtr(y), out_desc.get(),
                      GetBasePtr(out), ToCnnlDataType<T>());
  }
};

template <typename T>
class ElementwiseAddGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MLUDeviceContext>();
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    axis = (axis == -1 ? std::abs(x->dims().size() - y->dims().size()) : axis);

    MLUCnnlTensorDesc dout_desc(*dout);
    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      if (dx->dims() != dout->dims()) {
        std::vector<int> dst_dims_vec;
        std::vector<int> reduce_axes;
        auto src_dims = dx->dims();
        auto dout_dims = dout->dims();

        int src_axis = (src_dims.size() < dout_dims.size() ? axis : 0);
        for (int ax = 0; ax < dout_dims.size(); ++ax) {
          if ((ax < src_axis || ax >= src_axis + src_dims.size()) ||
              (dout_dims[ax] > 1 && src_dims[ax - src_axis] == 1)) {
            reduce_axes.push_back(ax);
          } else {
            dst_dims_vec.push_back(dout_dims[ax]);
          }
        }
        if (dst_dims_vec.size() == 0) {
          // x is scalar
          dst_dims_vec.push_back(1);
        }

        MLUCnnlReduceDesc reduction_desc(
            reduce_axes, CNNL_REDUCE_ADD, ToCnnlDataType<T>(),
            CNNL_NOT_PROPAGATE_NAN, CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES);
        MLUCnnlTensorDesc dx_desc(dst_dims_vec.size(), dst_dims_vec.data(),
                                  ToCnnlDataType<T>());
        MLUCnnl::Reduce(ctx, true /*need_workspace*/, reduction_desc.get(),
                        nullptr, dout_desc.get(), GetBasePtr(dout), 0, nullptr,
                        nullptr, dx_desc.get(), GetBasePtr(dx));
      } else {
        framework::TensorCopy(*dout, ctx.GetPlace(), dev_ctx, dx);
      }
    }
    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
      if (dy->dims() != dout->dims()) {
        std::vector<int> dst_dims_vec;
        std::vector<int> reduce_axes;
        auto src_dims = dy->dims();
        auto dout_dims = dout->dims();

        int src_axis = (src_dims.size() < dout_dims.size() ? axis : 0);
        for (int ax = 0; ax < dout_dims.size(); ++ax) {
          if ((ax < src_axis || ax >= src_axis + src_dims.size()) ||
              (dout_dims[ax] > 1 && src_dims[ax - src_axis] == 1)) {
            reduce_axes.push_back(ax);
          } else {
            dst_dims_vec.push_back(dout_dims[ax]);
          }
        }
        if (dst_dims_vec.size() == 0) {
          // y is scalar
          dst_dims_vec.push_back(1);
        }

        MLUCnnlReduceDesc reduction_desc(
            reduce_axes, CNNL_REDUCE_ADD, ToCnnlDataType<T>(),
            CNNL_NOT_PROPAGATE_NAN, CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES);
        MLUCnnlTensorDesc dy_desc(dst_dims_vec.size(), dst_dims_vec.data(),
                                  ToCnnlDataType<T>());
        MLUCnnl::Reduce(ctx, true /*need_workspace*/, reduction_desc.get(),
                        nullptr, dout_desc.get(), GetBasePtr(dout), 0, nullptr,
                        nullptr, dy_desc.get(), GetBasePtr(dy));
      } else {
        framework::TensorCopy(*dout, ctx.GetPlace(), dev_ctx, dy);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(elementwise_add, ops::ElementwiseAddMLUKernel<float>,
                       ops::ElementwiseAddMLUKernel<plat::float16>);
REGISTER_OP_MLU_KERNEL(elementwise_add_grad,
                       ops::ElementwiseAddGradMLUKernel<float>,
                       ops::ElementwiseAddGradMLUKernel<plat::float16>);
