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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename T>
class HuberLossMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = GetDevCtxFromCTX(ctx);
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* residual = ctx.Output<Tensor>("Residual");
    auto* out = ctx.Output<Tensor>("Out");
    auto delta = ctx.Attr<float>("delta");

    auto place = ctx.GetPlace();

    // compute y-x
    cnnlDataType_t data_type = ToCnnlDataType<T>();
    residual->mutable_data<T>(x->dims(), place);
    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlOpTensorDesc sub_op_desc(
        CNNL_OP_TENSOR_SUB, data_type, CNNL_NOT_PROPAGATE_NAN);
    MLUCnnl::OpTensor(ctx,
                      sub_op_desc.get(),
                      x_desc.get(),
                      GetBasePtr(y),
                      x_desc.get(),
                      GetBasePtr(x),
                      x_desc.get(),
                      GetBasePtr(residual),
                      data_type);

    // compute smoothl1loss
    out->mutable_data<T>(x->dims(), place);
    cnnlSmoothL1LossAlgorithm_t smoothl1_algo =
        CNNL_SMOOTHL1LOSS_REDUCTION_NONE;  // defines whether to do reduction
                                           // here
    MLUCnnl::SmoothL1LossForward(ctx,
                                 x_desc.get(),
                                 GetBasePtr(x),
                                 x_desc.get(), /* target has same shape as x */
                                 GetBasePtr(y),
                                 static_cast<float>(delta),
                                 smoothl1_algo,
                                 x_desc.get(), /* out has same shape as x */
                                 GetBasePtr(out));

    // compute multiply by delta
    framework::Tensor scale_tensor, bias_tensor;
    scale_tensor = ctx.AllocateTmpTensor<T, MLUDeviceContext>({1}, dev_ctx);
    bias_tensor = ctx.AllocateTmpTensor<T, MLUDeviceContext>({1}, dev_ctx);
    FillMLUTensorWithHostValue(ctx, static_cast<T>(delta), &scale_tensor);
    FillMLUTensorWithHostValue(ctx, static_cast<T>(0.f), &bias_tensor);
    const int axis = std::max(out->dims().size() - 1, 0);

    MLUCnnlTensorDesc scale_desc(scale_tensor);
    MLUCnnlTensorDesc bias_desc(bias_tensor);
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnl::Scale(ctx,
                   axis,
                   out_desc.get(),
                   GetBasePtr(out),
                   scale_desc.get(),
                   GetBasePtr(&scale_tensor),
                   bias_desc.get(),
                   GetBasePtr(&bias_tensor),
                   out_desc.get(),
                   GetBasePtr(out));
  }
};

template <typename T>
class HuberLossGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = GetDevCtxFromCTX(ctx);
    auto* residual = ctx.Input<Tensor>("Residual");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    auto delta = ctx.Attr<float>("delta");

    auto place = ctx.GetPlace();

    Tensor t_grad_rd;
    t_grad_rd =
        ctx.AllocateTmpTensor<T, MLUDeviceContext>(residual->dims(), dev_ctx);
    MLUCnnlTensorDesc t_grad_rd_desc(t_grad_rd);
    if (dx || dy) {
      Tensor t_zero;
      t_zero =
          ctx.AllocateTmpTensor<T, MLUDeviceContext>(residual->dims(), dev_ctx);
      FillMLUTensorWithHostValue(ctx, static_cast<T>(0.f), &t_zero);

      MLUCnnlTensorDesc residual_desc(*residual);
      MLUCnnlTensorDesc dout_desc(*dout);

      cnnlSmoothL1LossAlgorithm_t smoothl1_algo =
          CNNL_SMOOTHL1LOSS_REDUCTION_NONE;  // defines whether to do reduction
                                             // here
      MLUCnnl::SmoothL1LossBackward(ctx,
                                    residual_desc.get(),
                                    GetBasePtr(residual),
                                    residual_desc.get(),
                                    GetBasePtr(&t_zero),
                                    dout_desc.get(),
                                    GetBasePtr(dout),
                                    static_cast<float>(delta),
                                    smoothl1_algo,
                                    t_grad_rd_desc.get(),
                                    GetBasePtr(&t_grad_rd));
    }
    // compute multiply by delta
    framework::Tensor scale_tensor, bias_tensor;
    scale_tensor = ctx.AllocateTmpTensor<T, MLUDeviceContext>({1}, dev_ctx);
    bias_tensor = ctx.AllocateTmpTensor<T, MLUDeviceContext>({1}, dev_ctx);

    FillMLUTensorWithHostValue(ctx, static_cast<T>(0.f), &bias_tensor);
    const int axis = std::max(t_grad_rd.dims().size() - 1, 0);

    MLUCnnlTensorDesc scale_desc(scale_tensor);
    MLUCnnlTensorDesc bias_desc(bias_tensor);

    if (dx) {
      dx->mutable_data<T>(place);
      FillMLUTensorWithHostValue(ctx, static_cast<T>(-delta), &scale_tensor);
      MLUCnnlTensorDesc out_desc(*dx);
      MLUCnnl::Scale(ctx,
                     axis,
                     t_grad_rd_desc.get(),
                     GetBasePtr(&t_grad_rd),
                     scale_desc.get(),
                     GetBasePtr(&scale_tensor),
                     bias_desc.get(),
                     GetBasePtr(&bias_tensor),
                     out_desc.get(),
                     GetBasePtr(dx));
    }
    if (dy) {
      dy->mutable_data<T>(place);
      FillMLUTensorWithHostValue(ctx, static_cast<T>(delta), &scale_tensor);
      MLUCnnlTensorDesc out_desc(*dy);
      MLUCnnl::Scale(ctx,
                     axis,
                     t_grad_rd_desc.get(),
                     GetBasePtr(&t_grad_rd),
                     scale_desc.get(),
                     GetBasePtr(&scale_tensor),
                     bias_desc.get(),
                     GetBasePtr(&bias_tensor),
                     out_desc.get(),
                     GetBasePtr(dy));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(huber_loss,
                       ops::HuberLossMLUKernel<float>,
                       ops::HuberLossMLUKernel<plat::float16>);
REGISTER_OP_MLU_KERNEL(huber_loss_grad,
                       ops::HuberLossGradMLUKernel<float>,
                       ops::HuberLossGradMLUKernel<plat::float16>);
