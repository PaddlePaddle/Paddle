/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include <memory>
#include <string>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/fluid/platform/device/mlu/device_context.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <cnnlActivationMode_t act_mode, typename T>
class ActivationMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 1.0f;

    output->mutable_data<T>(ctx.GetPlace());

    MLUCnnlActivationDesc act_desc(act_mode, alpha);
    MLUCnnlTensorDesc input_desc(
        *input, CNNL_LAYOUT_ARRAY,
        ToCnnlDataType(framework::TransToProtoVarType(input->dtype())));
    MLUCnnlTensorDesc output_desc(
        *output, CNNL_LAYOUT_ARRAY,
        ToCnnlDataType(framework::TransToProtoVarType(output->dtype())));

    MLUCnnl::Active(ctx, act_desc.get(), input_desc.get(),
                    reinterpret_cast<const void*>(input->data<T>()),
                    output_desc.get(),
                    reinterpret_cast<void*>(output->data<T>()));
  }
};

template <cnnlActivationMode_t act_mode, typename T>
class ActivationGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 1.0f;

    dx->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc dout_desc(
        *dout, CNNL_LAYOUT_ARRAY,
        ToCnnlDataType(framework::TransToProtoVarType(dout->dtype())));
    MLUCnnlTensorDesc out_desc(
        *out, CNNL_LAYOUT_ARRAY,
        ToCnnlDataType(framework::TransToProtoVarType(out->dtype())));
    MLUCnnlTensorDesc dx_desc(
        *dx, CNNL_LAYOUT_ARRAY,
        ToCnnlDataType(framework::TransToProtoVarType(dx->dtype())));
    MLUCnnlActivationDesc act_desc(act_mode, alpha);
    MLUCnnl::ActiveGrad(
        ctx, act_desc.get(), nullptr, nullptr, nullptr, nullptr,
        dout_desc.get(), reinterpret_cast<const void*>(dout->data<T>()),
        out_desc.get(), reinterpret_cast<const void*>(out->data<T>()),
        dx_desc.get(), reinterpret_cast<void*>(dx->data<T>()));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(
    relu, ops::ActivationMLUKernel<CNNL_ACTIVATION_RELU, float>,
    ops::ActivationMLUKernel<CNNL_ACTIVATION_RELU, paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(
    relu_grad, ops::ActivationGradMLUKernel<CNNL_ACTIVATION_RELU, float>,
    ops::ActivationGradMLUKernel<CNNL_ACTIVATION_RELU,
                                 paddle::platform::float16>);
