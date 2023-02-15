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

#include "paddle/fluid/operators/elementwise/elementwise_div_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_mlu.h"

namespace paddle {
namespace operators {

template <typename T>
class ElementwiseDivMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    MLUBinaryOp<DIV, T>(ctx);
  }
};

template <typename T>
class ElementwiseDivGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<phi::DenseTensor>("Out");
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* y = ctx.Input<phi::DenseTensor>("Y");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<phi::DenseTensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");

    const auto& x_dims = x->dims();
    const auto& y_dims = y->dims();
    axis = (axis < 0 ? (std::abs(x_dims.size() - y_dims.size()) + axis + 1)
                     : axis);
    int max_dim = std::max(x_dims.size(), y_dims.size());
    std::vector<int> x_dims_array(max_dim);
    std::vector<int> y_dims_array(max_dim);
    std::vector<int> out_dims_array(max_dim);
    GetBroadcastDimsArrays(x_dims,
                           y_dims,
                           x_dims_array.data(),
                           y_dims_array.data(),
                           out_dims_array.data(),
                           max_dim,
                           axis);

    MLUCnnlTensorDesc x_desc(max_dim, x_dims_array.data(), ToCnnlDataType<T>());
    MLUCnnlTensorDesc y_desc(max_dim, y_dims_array.data(), ToCnnlDataType<T>());
    MLUCnnlTensorDesc dout_desc(*dout);
    MLUCnnlOpTensorDesc mul_op_desc(
        CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

    // compute dout/y == 1/y * dout
    phi::DenseTensor dout_div_y(dout->dtype());
    dout_div_y.Resize(dout->dims());
    dout_div_y.mutable_data<T>(ctx.GetPlace());
    MLUBinary<DIV>(ctx,
                   CNNL_COMPUTATION_HIGH_PRECISION,
                   dout_desc.get(),
                   GetBasePtr(dout),
                   y_desc.get(),
                   GetBasePtr(y),
                   dout_desc.get(),
                   GetBasePtr(&dout_div_y));

    if (dx) {
      // compute dx = dout/y = 1/y * dout
      if (dx->dims() != dout->dims()) {
        dx->mutable_data<T>(ctx.GetPlace());

        std::vector<int> reduce_axes;
        GetReduceAxes(axis, dout_div_y.dims(), dx->dims(), &reduce_axes);
        MLUCnnlReduceDesc reduction_desc(reduce_axes,
                                         CNNL_REDUCE_ADD,
                                         ToCnnlDataType<T>(),
                                         CNNL_NOT_PROPAGATE_NAN,
                                         CNNL_REDUCE_NO_INDICES,
                                         CNNL_32BIT_INDICES);
        MLUCnnlTensorDesc dx_desc(*dx);
        MLUCnnl::Reduce(ctx,
                        true /*need_workspace*/,
                        reduction_desc.get(),
                        nullptr,
                        dout_desc.get(),
                        GetBasePtr(&dout_div_y),
                        0,
                        nullptr,
                        nullptr,
                        dx_desc.get(),
                        GetBasePtr(dx));
      } else {
        dx->ShareDataWith(dout_div_y);
      }
    }

    if (dy) {
      // compute dy = -out * (dout/y) = -out/y * dout
      phi::DenseTensor neg_out(out->type());
      neg_out.mutable_data<T>(out->dims(), ctx.GetPlace());

      MLUCnnlTensorDesc out_desc(*out);
      MLUUnary<NEG>(ctx,
                    CNNL_COMPUTATION_HIGH_PRECISION,
                    out_desc.get(),
                    GetBasePtr(out),
                    out_desc.get(),
                    GetBasePtr(&neg_out));

      phi::DenseTensor dy_temp(y->dtype());
      dy_temp.Resize(dout->dims());
      dy_temp.mutable_data<T>(ctx.GetPlace());

      MLUCnnl::OpTensor(ctx,
                        mul_op_desc.get(),
                        dout_desc.get(),
                        GetBasePtr(&neg_out),
                        dout_desc.get(),
                        GetBasePtr(&dout_div_y),
                        dout_desc.get(),
                        GetBasePtr(&dy_temp),
                        ToCnnlDataType<T>());

      if (dy->dims() != dout->dims()) {
        dy->mutable_data<T>(ctx.GetPlace());

        std::vector<int> reduce_axes;
        GetReduceAxes(axis, dy_temp.dims(), dy->dims(), &reduce_axes);
        MLUCnnlReduceDesc reduction_desc(reduce_axes,
                                         CNNL_REDUCE_ADD,
                                         ToCnnlDataType<T>(),
                                         CNNL_NOT_PROPAGATE_NAN,
                                         CNNL_REDUCE_NO_INDICES,
                                         CNNL_32BIT_INDICES);
        MLUCnnlTensorDesc dy_desc(*dy);
        MLUCnnl::Reduce(ctx,
                        true /*need_workspace*/,
                        reduction_desc.get(),
                        nullptr,
                        dout_desc.get(),
                        GetBasePtr(&dy_temp),
                        0,
                        nullptr,
                        nullptr,
                        dy_desc.get(),
                        GetBasePtr(dy));
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

REGISTER_OP_MLU_KERNEL(elementwise_div,
                       ops::ElementwiseDivMLUKernel<int>,
                       ops::ElementwiseDivMLUKernel<float>,
                       ops::ElementwiseDivMLUKernel<plat::float16>);

REGISTER_OP_MLU_KERNEL(elementwise_div_grad,
                       ops::ElementwiseDivGradMLUKernel<int>,
                       ops::ElementwiseDivGradMLUKernel<float>,
                       ops::ElementwiseDivGradMLUKernel<plat::float16>);
