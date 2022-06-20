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

#include <memory>
#include <string>

#include "paddle/fluid/operators/elementwise/elementwise_mlu.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class ElementwiseMinMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    MLUBinaryOp<MINIMUM, T>(ctx);
  }
};

template <typename T>
class ElementwiseMinGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
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

    // mask = LessEqual(x, y)
    Tensor mask(x->dtype());
    mask.Resize(phi::make_ddim(out_dims_array));
    mask.mutable_data<T>(ctx.GetPlace());

    cnnlDataType_t data_type = ToCnnlDataType<T>();
    MLUCnnlTensorDesc x_desc(max_dim, x_dims_array.data(), data_type);
    MLUCnnlTensorDesc y_desc(max_dim, y_dims_array.data(), data_type);
    MLUCnnlTensorDesc mask_desc(max_dim, out_dims_array.data(), data_type);
    MLUCnnl::Logic(ctx, CNNL_LOGIC_OP_LE, x_desc.get(), GetBasePtr(x),
                   y_desc.get(), GetBasePtr(y), mask_desc.get(),
                   GetBasePtr(&mask));

    // dx = Mul(dz, mask)
    Tensor dx_temp(x->dtype());
    dx_temp.Resize(dout->dims());
    dx_temp.mutable_data<T>(ctx.GetPlace());
    MLUCnnlTensorDesc dout_desc(*dout);
    MLUCnnlOpTensorDesc mul_op_desc(CNNL_OP_TENSOR_MUL, data_type,
                                    CNNL_NOT_PROPAGATE_NAN);
    MLUCnnl::OpTensor(ctx, mul_op_desc.get(), dout_desc.get(), GetBasePtr(dout),
                      dout_desc.get(), GetBasePtr(&mask), dout_desc.get(),
                      GetBasePtr(&dx_temp), data_type);

    // dy = Sub(dz, dx)
    Tensor dy_temp(y->dtype());
    dy_temp.Resize(dout->dims());
    dy_temp.mutable_data<T>(ctx.GetPlace());
    MLUCnnlOpTensorDesc sub_op_desc(CNNL_OP_TENSOR_SUB, data_type,
                                    CNNL_NOT_PROPAGATE_NAN);
    MLUCnnl::OpTensor(ctx, sub_op_desc.get(), dout_desc.get(), GetBasePtr(dout),
                      dout_desc.get(), GetBasePtr(&dx_temp), dout_desc.get(),
                      GetBasePtr(&dy_temp), data_type);

    if (dx) {
      if (dx->dims() != dout->dims()) {
        dx->mutable_data<T>(ctx.GetPlace());
        std::vector<int> reduce_axes;
        GetReduceAxes(axis, dx_temp.dims(), dx->dims(), &reduce_axes);
        MLUCnnlReduceDesc reduction_desc(
            reduce_axes, CNNL_REDUCE_ADD, data_type, CNNL_NOT_PROPAGATE_NAN,
            CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES);
        MLUCnnlTensorDesc dx_desc(*dx);
        MLUCnnl::Reduce(ctx, true /*need_workspace*/, reduction_desc.get(),
                        nullptr, dout_desc.get(), GetBasePtr(&dx_temp), 0,
                        nullptr, nullptr, dx_desc.get(), GetBasePtr(dx));
      } else {
        dx->ShareDataWith(dx_temp);
      }
    }

    if (dy) {
      if (dy->dims() != dout->dims()) {
        dy->mutable_data<T>(ctx.GetPlace());
        std::vector<int> reduce_axes;
        GetReduceAxes(axis, dy_temp.dims(), dy->dims(), &reduce_axes);
        MLUCnnlReduceDesc reduction_desc(
            reduce_axes, CNNL_REDUCE_ADD, data_type, CNNL_NOT_PROPAGATE_NAN,
            CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES);
        MLUCnnlTensorDesc dy_desc(*dy);
        MLUCnnl::Reduce(ctx, true /*need_workspace*/, reduction_desc.get(),
                        nullptr, dout_desc.get(), GetBasePtr(&dy_temp), 0,
                        nullptr, nullptr, dy_desc.get(), GetBasePtr(dy));
      } else {
        dy->ShareDataWith(dy_temp);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(elementwise_min, ops::ElementwiseMinMLUKernel<int>,
                       ops::ElementwiseMinMLUKernel<float>,
                       ops::ElementwiseMinMLUKernel<plat::float16>);

REGISTER_OP_MLU_KERNEL(elementwise_min_grad,
                       ops::ElementwiseMinGradMLUKernel<int>,
                       ops::ElementwiseMinGradMLUKernel<float>,
                       ops::ElementwiseMinGradMLUKernel<plat::float16>);
