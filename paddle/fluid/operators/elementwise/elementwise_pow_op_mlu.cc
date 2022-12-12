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

#include "paddle/fluid/operators/elementwise/elementwise_mlu.h"
#include "paddle/fluid/operators/elementwise/elementwise_op.h"

namespace paddle {
namespace operators {

template <typename T>
class ElementwisePowMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    MLUBinaryOp<POW, T>(ctx);
  }
};

template <typename T>
class ElementwisePowGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* y = ctx.Input<phi::DenseTensor>("Y");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<phi::DenseTensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    auto place = ctx.GetPlace();

    auto x_dims = x->dims();
    auto y_dims = y->dims();
    axis =
        (axis < 0 ? std::abs(x_dims.size() - y_dims.size()) + axis + 1 : axis);

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
    cnnlDataType_t data_type = ToCnnlDataType<T>();
    MLUCnnlTensorDesc x_desc(max_dim, x_dims_array.data(), data_type);
    MLUCnnlTensorDesc y_desc(max_dim, y_dims_array.data(), data_type);
    MLUCnnlTensorDesc out_desc(max_dim, out_dims_array.data(), data_type);

    auto dout_dims = dout->dims();
    if (dx) {
      // dx = dout * y * pow(x, y - 1);
      phi::DenseTensor one_dx(y->type());
      one_dx.mutable_data<T>(phi::make_ddim(y_dims_array), place);
      FillMLUTensorWithHostValue(ctx, static_cast<T>(1), &one_dx);

      phi::DenseTensor sub_dx(y->type());
      sub_dx.mutable_data<T>(phi::make_ddim(y_dims_array), place);
      MLUCnnlOpTensorDesc op_tensor_desc(
          CNNL_OP_TENSOR_SUB, data_type, CNNL_NOT_PROPAGATE_NAN);
      MLUCnnl::OpTensor(ctx,
                        op_tensor_desc.get(),
                        y_desc.get(),
                        GetBasePtr(y),
                        y_desc.get(),
                        GetBasePtr(&one_dx),
                        y_desc.get(),
                        GetBasePtr(&sub_dx),
                        data_type);

      phi::DenseTensor tmp_dx(x->type());
      tmp_dx.mutable_data<T>(phi::make_ddim(out_dims_array), place);
      MLUCnnl::Pow(ctx,
                   CNNL_COMPUTATION_HIGH_PRECISION,
                   x_desc.get(),
                   GetBasePtr(x),
                   y_desc.get(),
                   GetBasePtr(&sub_dx),
                   out_desc.get(),
                   GetBasePtr(&tmp_dx));

      MLUCnnl::MulAx(ctx,
                     y_desc.get(),
                     GetBasePtr(y),
                     out_desc.get(),
                     GetBasePtr(&tmp_dx));
      MLUCnnl::MulAx(ctx,
                     out_desc.get(),
                     GetBasePtr(dout),
                     out_desc.get(),
                     GetBasePtr(&tmp_dx));

      if (x_dims != dout_dims) {
        dx->mutable_data<T>(place);
        std::vector<int> reduce_axes;
        GetReduceAxes(axis, dout_dims, x_dims, &reduce_axes);
        if (!reduce_axes.empty()) {
          MLUCnnlReduceDesc reduction_desc(reduce_axes,
                                           CNNL_REDUCE_ADD,
                                           data_type,
                                           CNNL_NOT_PROPAGATE_NAN,
                                           CNNL_REDUCE_NO_INDICES,
                                           CNNL_32BIT_INDICES);
          MLUCnnlTensorDesc dx_desc(*dx);
          MLUCnnl::Reduce(ctx,
                          true /*need_workspace*/,
                          reduction_desc.get(),
                          nullptr,
                          out_desc.get(),
                          GetBasePtr(&tmp_dx),
                          0,
                          nullptr,
                          nullptr,
                          dx_desc.get(),
                          GetBasePtr(dx));
        }
      } else {
        dx->ShareDataWith(tmp_dx);
      }
    }
    if (dy) {
      // dy = dout * log(x) * pow(x, y)
      phi::DenseTensor tmp_dy(y->type());
      tmp_dy.mutable_data<T>(phi::make_ddim(out_dims_array), place);
      MLUCnnl::Pow(ctx,
                   CNNL_COMPUTATION_HIGH_PRECISION,
                   x_desc.get(),
                   GetBasePtr(x),
                   y_desc.get(),
                   GetBasePtr(y),
                   out_desc.get(),
                   GetBasePtr(&tmp_dy));

      phi::DenseTensor log_x(x->type());
      log_x.mutable_data<T>(x->dims(), place);
      MLUCnnl::Log(ctx,
                   CNNL_COMPUTATION_HIGH_PRECISION,
                   CNNL_LOG_E,
                   x_desc.get(),
                   GetBasePtr(x),
                   x_desc.get(),
                   GetBasePtr(&log_x));
      MLUCnnl::MulAx(ctx,
                     x_desc.get(),
                     GetBasePtr(&log_x),
                     out_desc.get(),
                     GetBasePtr(&tmp_dy));
      MLUCnnl::MulAx(ctx,
                     out_desc.get(),
                     GetBasePtr(dout),
                     out_desc.get(),
                     GetBasePtr(&tmp_dy));

      if (y_dims != dout_dims) {
        dy->mutable_data<T>(place);
        std::vector<int> reduce_axes;
        GetReduceAxes(axis, dout_dims, y_dims, &reduce_axes);
        if (!reduce_axes.empty()) {
          MLUCnnlReduceDesc reduction_desc(reduce_axes,
                                           CNNL_REDUCE_ADD,
                                           data_type,
                                           CNNL_NOT_PROPAGATE_NAN,
                                           CNNL_REDUCE_NO_INDICES,
                                           CNNL_32BIT_INDICES);
          MLUCnnlTensorDesc dy_desc(*dy);
          MLUCnnl::Reduce(ctx,
                          true /*need_workspace*/,
                          reduction_desc.get(),
                          nullptr,
                          out_desc.get(),
                          GetBasePtr(&tmp_dy),
                          0,
                          nullptr,
                          nullptr,
                          dy_desc.get(),
                          GetBasePtr(dy));
        }
      } else {
        dy->ShareDataWith(tmp_dy);
      }
    }
    if (!dx && !dy) {
      PADDLE_THROW(platform::errors::Unavailable(
          "Not support all outputs to be empty."));
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(elementwise_pow,
                       ops::ElementwisePowMLUKernel<plat::float16>,
                       ops::ElementwisePowMLUKernel<float>);

REGISTER_OP_MLU_KERNEL(elementwise_pow_grad,
                       ops::ElementwisePowGradMLUKernel<plat::float16>,
                       ops::ElementwisePowGradMLUKernel<float>);
