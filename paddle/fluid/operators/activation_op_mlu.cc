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

#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <cnnlActivationMode_t act_mode, typename T>
class ActivationMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* output = ctx.Output<phi::DenseTensor>("Out");
    float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 1.0f;

    output->mutable_data<T>(ctx.GetPlace());

    MLUCnnlActivationDesc act_desc(act_mode, alpha);
    MLUCnnlTensorDesc input_desc(*input);
    MLUCnnlTensorDesc output_desc(*output);

    MLUCnnl::Active(ctx,
                    act_desc.get(),
                    input_desc.get(),
                    GetBasePtr(input),
                    output_desc.get(),
                    GetBasePtr(output));
  }
};

// For gelu, leaky_relu
template <cnnlActivationMode_t act_mode, typename T>
class ActivationGradMLUKernelV1 : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 1.0f;

    dx->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc dout_desc(*dout);
    MLUCnnlTensorDesc dx_desc(*dx);
    MLUCnnlActivationDesc act_desc(act_mode, alpha);
    MLUCnnl::ActiveGrad(ctx,
                        act_desc.get(),
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        dout_desc.get(),
                        GetBasePtr(dout),
                        x_desc.get(),
                        GetBasePtr(x),
                        dx_desc.get(),
                        GetBasePtr(dx));
  }
};

// For tanh, sigmoid
template <cnnlActivationMode_t act_mode, typename T>
class ActivationGradMLUKernelV2 : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<phi::DenseTensor>("Out");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 1.0f;

    dx->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnlTensorDesc dout_desc(*dout);
    MLUCnnlTensorDesc dx_desc(*dx);
    MLUCnnlActivationDesc act_desc(act_mode, alpha);
    MLUCnnl::ActiveGrad(ctx,
                        act_desc.get(),
                        nullptr,
                        nullptr,
                        out_desc.get(),
                        GetBasePtr(out),
                        dout_desc.get(),
                        GetBasePtr(dout),
                        nullptr,
                        nullptr,
                        dx_desc.get(),
                        GetBasePtr(dx));
  }
};

// For relu, relu6
template <cnnlActivationMode_t act_mode, typename T>
class ActivationGradMLUKernelV3 : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<phi::DenseTensor>("Out");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 1.0f;

    dx->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnlTensorDesc dout_desc(*dout);
    MLUCnnlTensorDesc dx_desc(*dx);
    MLUCnnlActivationDesc act_desc(act_mode, alpha);
    MLUCnnl::ActiveGrad(ctx,
                        act_desc.get(),
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        dout_desc.get(),
                        GetBasePtr(dout),
                        out_desc.get(),
                        GetBasePtr(out),
                        dx_desc.get(),
                        GetBasePtr(dx));
  }
};

// For sqrt
template <typename T>
class SqrtMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    auto place = ctx.GetPlace();

    out->mutable_data<T>(place);

    MLUCnnlTensorDesc input_desc(*x);
    MLUCnnlTensorDesc output_desc(*out);

    cnnlComputationPreference_t prefer = CNNL_COMPUTATION_FAST;
    MLUCnnl::Sqrt(ctx,
                  prefer,
                  input_desc.get(),
                  GetBasePtr(x),
                  output_desc.get(),
                  GetBasePtr(out));
  }
};

template <typename T>
class SqrtGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<phi::DenseTensor>("Out");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto place = ctx.GetPlace();

    dx->mutable_data<T>(place);

    MLUCnnlTensorDesc data_desc(*out);
    MLUCnnl::SqrtGrad(ctx,
                      data_desc.get(),
                      GetBasePtr(out),
                      GetBasePtr(dout),
                      GetBasePtr(dx));
  }
};

// CNNL_LOG_E = 0,
// CNNL_LOG_2 = 1,
// CNNL_LOG_10 = 2,
template <cnnlLogBase_t Log_base, typename T>
class LogMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* output = ctx.Output<phi::DenseTensor>("Out");
    output->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc input_desc(*input);
    MLUCnnlTensorDesc output_desc(*output);
    cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;

    MLUCnnl::Log(ctx,
                 prefer,
                 Log_base,
                 input_desc.get(),
                 GetBasePtr(input),
                 output_desc.get(),
                 GetBasePtr(output));
  }
};

template <typename T>
class ExpMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* output = ctx.Output<phi::DenseTensor>("Out");
    output->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc input_desc(*input);
    MLUCnnlTensorDesc output_desc(*output);
    cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;

    MLUCnnl::Exp(ctx,
                 prefer,
                 input_desc.get(),
                 GetBasePtr(input),
                 output_desc.get(),
                 GetBasePtr(output));
  }
};

template <typename T>
class ExpGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<phi::DenseTensor>("Out");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());
    MLUCnnlTensorDesc dout_desc(*dout);
    MLUCnnlTensorDesc dx_desc(*dx);
    MLUCnnlTensorDesc out_desc(*out);

    MLUCnnlOpTensorDesc op_tensor_desc(
        CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

    MLUCnnl::OpTensor(ctx,
                      op_tensor_desc.get(),
                      dout_desc.get(),
                      GetBasePtr(dout),
                      out_desc.get(),
                      GetBasePtr(out),
                      dx_desc.get(),
                      GetBasePtr(dx),
                      ToCnnlDataType<T>());
  }
};

template <typename T>
class HardSwishMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* output = ctx.Output<phi::DenseTensor>("Out");
    output->mutable_data<T>(ctx.GetPlace());
    float threshold = ctx.Attr<float>("threshold");
    float scale = ctx.Attr<float>("scale");
    float offset = ctx.Attr<float>("offset");
    PADDLE_ENFORCE_EQ(threshold,
                      6.0f,
                      platform::errors::External(
                          "Not support threshold [%f] in MLU", threshold));
    PADDLE_ENFORCE_EQ(
        scale,
        6.0f,
        platform::errors::External("Not support scale [%f] in MLU", scale));
    PADDLE_ENFORCE_EQ(
        offset,
        3.0f,
        platform::errors::External("Not support offset [%f] in MLU", offset));

    MLUCnnlActivationDesc act_desc(CNNL_ACTIVATION_HARDSWISH,
                                   1.0f /*ceof useless*/);
    MLUCnnlTensorDesc input_desc(*input);
    MLUCnnlTensorDesc output_desc(*output);

    MLUCnnl::Active(ctx,
                    act_desc.get(),
                    input_desc.get(),
                    GetBasePtr(input),
                    output_desc.get(),
                    GetBasePtr(output));
  }
};

template <typename T>
class HardSwishGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    float threshold = ctx.Attr<float>("threshold");
    float scale = ctx.Attr<float>("scale");
    float offset = ctx.Attr<float>("offset");
    PADDLE_ENFORCE_EQ(threshold,
                      6.0f,
                      platform::errors::External(
                          "Not support threshold [%f] in MLU", threshold));
    PADDLE_ENFORCE_EQ(
        scale,
        6.0f,
        platform::errors::External("Not support scale [%f] in MLU", scale));
    PADDLE_ENFORCE_EQ(
        offset,
        3.0f,
        platform::errors::External("Not support offset [%f] in MLU", offset));
    auto* out = ctx.Input<phi::DenseTensor>("X");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));

    dx->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnlTensorDesc dout_desc(*dout);
    MLUCnnlTensorDesc dx_desc(*dx);
    MLUCnnlActivationDesc act_desc(CNNL_ACTIVATION_HARDSWISH,
                                   1.0f /*ceof useless*/);
    MLUCnnl::ActiveGrad(ctx,
                        act_desc.get(),
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        dout_desc.get(),
                        GetBasePtr(dout),
                        out_desc.get(),
                        GetBasePtr(out),
                        dx_desc.get(),
                        GetBasePtr(dx));
  }
};

template <typename T>
class HardSigmoidMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* output = ctx.Output<phi::DenseTensor>("Out");
    float slope = ctx.Attr<float>("slope");
    float offset = ctx.Attr<float>("offset");
    output->mutable_data<T>(ctx.GetPlace());

    MLUCnnlActivationDesc act_desc(CNNL_ACTIVATION_HARDSIGMOID,
                                   1.0f /*ceof useless*/,
                                   1.0f /*sliced_dim useless*/,
                                   slope,
                                   offset);
    MLUCnnlTensorDesc input_desc(*input);
    MLUCnnlTensorDesc output_desc(*output);

    MLUCnnl::Active(ctx,
                    act_desc.get(),
                    input_desc.get(),
                    GetBasePtr(input),
                    output_desc.get(),
                    GetBasePtr(output));
  }
};

template <typename T>
class HardSigmoidGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    float slope = ctx.Attr<float>("slope");
    float offset = ctx.Attr<float>("offset");
    dx->mutable_data<T>(ctx.GetPlace());

    MLUCnnlActivationDesc act_desc(CNNL_ACTIVATION_HARDSIGMOID,
                                   1.0f /*ceof useless*/,
                                   1.0f /*sliced_dim useless*/,
                                   slope,
                                   offset);
    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc dout_desc(*dout);
    MLUCnnlTensorDesc dx_desc(*dx);
    MLUCnnl::ActiveGrad(ctx,
                        act_desc.get(),
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        dout_desc.get(),
                        GetBasePtr(dout),
                        x_desc.get(),
                        GetBasePtr(x),
                        dx_desc.get(),
                        GetBasePtr(dx));
  }
};

template <typename T>
class FloorMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* output = ctx.Output<phi::DenseTensor>("Out");
    output->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc input_desc(*input);
    MLUCnnlTensorDesc output_desc(*output);

    MLUCnnl::Floor(ctx,
                   input_desc.get(),
                   GetBasePtr(input),
                   output_desc.get(),
                   GetBasePtr(output));
  }
};

template <typename DeviceContext, typename T>
class ReciprocalMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    auto place = ctx.GetPlace();
    out->mutable_data<T>(place);
    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnl::Reciprocal(
        ctx, x_desc.get(), GetBasePtr(x), out_desc.get(), GetBasePtr(out));
  }
};

template <typename DeviceContext, typename T>
class ReciprocalGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<phi::DenseTensor>("Out");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto place = ctx.GetPlace();
    dx->mutable_data<T>(place);
    phi::DenseTensor square_out;
    square_out.Resize(out->dims());
    square_out.mutable_data<T>(place);
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnlTensorDesc dout_desc(*dout);
    MLUCnnlTensorDesc dx_desc(*dx);
    MLUCnnlTensorDesc square_out_desc(square_out);
    MLUCnnl::Square(ctx,
                    out_desc.get(),
                    GetBasePtr(out),
                    square_out_desc.get(),
                    GetBasePtr(&square_out));
    cnnlOpTensorDesc_t op_tensor_op = CNNL_OP_TENSOR_MUL;
    cnnlDataType_t op_tensor_comp_type = CNNL_DTYPE_FLOAT;
    cnnlNanPropagation_t op_tensor_nan_opt = CNNL_NOT_PROPAGATE_NAN;
    MLUCnnlOpTensorDesc op_tensor_desc(
        op_tensor_op, op_tensor_comp_type, op_tensor_nan_opt);
    float alpha1_float = -1;
    float alpha2_float = 1;
    float beta_float = 0;
    MLUCnnl::OpTensor(ctx,
                      op_tensor_desc.get(),
                      dout_desc.get(),
                      GetBasePtr(dout),
                      square_out_desc.get(),
                      GetBasePtr(&square_out),
                      dx_desc.get(),
                      GetBasePtr(dx),
                      op_tensor_comp_type,
                      alpha1_float,
                      alpha2_float,
                      beta_float);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

// reciprocal
REGISTER_OP_MLU_KERNEL(
    reciprocal,
    ops::ReciprocalMLUKernel<paddle::platform::MLUDeviceContext, float>,
    ops::ReciprocalMLUKernel<paddle::platform::MLUDeviceContext,
                             paddle::platform::float16>);

REGISTER_OP_MLU_KERNEL(
    reciprocal_grad,
    ops::ReciprocalGradMLUKernel<paddle::platform::MLUDeviceContext, float>,
    ops::ReciprocalGradMLUKernel<paddle::platform::MLUDeviceContext,
                                 paddle::platform::float16>);
// relu
REGISTER_OP_MLU_KERNEL(
    relu,
    ops::ActivationMLUKernel<CNNL_ACTIVATION_RELU, float>,
    ops::ActivationMLUKernel<CNNL_ACTIVATION_RELU, paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(
    relu_grad,
    ops::ActivationGradMLUKernelV3<CNNL_ACTIVATION_RELU, float>,
    ops::ActivationGradMLUKernelV3<CNNL_ACTIVATION_RELU,
                                   paddle::platform::float16>);

// relu6
REGISTER_OP_MLU_KERNEL(
    relu6,
    ops::ActivationMLUKernel<CNNL_ACTIVATION_RELU6, float>,
    ops::ActivationMLUKernel<CNNL_ACTIVATION_RELU6, paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(
    relu6_grad,
    ops::ActivationGradMLUKernelV3<CNNL_ACTIVATION_RELU6, float>,
    ops::ActivationGradMLUKernelV3<CNNL_ACTIVATION_RELU6,
                                   paddle::platform::float16>);

// sigmoid
REGISTER_OP_MLU_KERNEL(sigmoid,
                       ops::ActivationMLUKernel<CNNL_ACTIVATION_SIGMOID, float>,
                       ops::ActivationMLUKernel<CNNL_ACTIVATION_SIGMOID,
                                                paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(
    sigmoid_grad,
    ops::ActivationGradMLUKernelV2<CNNL_ACTIVATION_SIGMOID, float>,
    ops::ActivationGradMLUKernelV2<CNNL_ACTIVATION_SIGMOID,
                                   paddle::platform::float16>);

// tanh
REGISTER_OP_MLU_KERNEL(
    tanh,
    ops::ActivationMLUKernel<CNNL_ACTIVATION_TANH, float>,
    ops::ActivationMLUKernel<CNNL_ACTIVATION_TANH, paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(
    tanh_grad,
    ops::ActivationGradMLUKernelV2<CNNL_ACTIVATION_TANH, float>,
    ops::ActivationGradMLUKernelV2<CNNL_ACTIVATION_TANH,
                                   paddle::platform::float16>);

// gelu
REGISTER_OP_MLU_KERNEL(
    gelu,
    ops::ActivationMLUKernel<CNNL_ACTIVATION_GELU, float>,
    ops::ActivationMLUKernel<CNNL_ACTIVATION_GELU, paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(
    gelu_grad,
    ops::ActivationGradMLUKernelV1<CNNL_ACTIVATION_GELU, float>,
    ops::ActivationGradMLUKernelV1<CNNL_ACTIVATION_GELU,
                                   paddle::platform::float16>);

// leaky_relu
REGISTER_OP_MLU_KERNEL(
    leaky_relu,
    ops::ActivationMLUKernel<CNNL_ACTIVATION_LEAKYRELU, float>,
    ops::ActivationMLUKernel<CNNL_ACTIVATION_LEAKYRELU,
                             paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(
    leaky_relu_grad,
    ops::ActivationGradMLUKernelV1<CNNL_ACTIVATION_LEAKYRELU, float>,
    ops::ActivationGradMLUKernelV1<CNNL_ACTIVATION_LEAKYRELU,
                                   paddle::platform::float16>);

// sqrt
REGISTER_OP_MLU_KERNEL(sqrt,
                       ops::SqrtMLUKernel<float>,
                       ops::SqrtMLUKernel<paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(sqrt_grad,
                       ops::SqrtGradMLUKernel<float>,
                       ops::SqrtGradMLUKernel<paddle::platform::float16>);

// log log2 log10
REGISTER_OP_MLU_KERNEL(
    log,
    ops::LogMLUKernel<CNNL_LOG_E, float>,
    ops::LogMLUKernel<CNNL_LOG_E, paddle::platform::float16>);

REGISTER_OP_MLU_KERNEL(
    log2,
    ops::LogMLUKernel<CNNL_LOG_2, float>,
    ops::LogMLUKernel<CNNL_LOG_2, paddle::platform::float16>);

REGISTER_OP_MLU_KERNEL(
    log10,
    ops::LogMLUKernel<CNNL_LOG_10, float>,
    ops::LogMLUKernel<CNNL_LOG_10, paddle::platform::float16>);

REGISTER_OP_MLU_KERNEL(exp,
                       ops::ExpMLUKernel<float>,
                       ops::ExpMLUKernel<paddle::platform::float16>);

REGISTER_OP_MLU_KERNEL(exp_grad,
                       ops::ExpGradMLUKernel<float>,
                       ops::ExpGradMLUKernel<paddle::platform::float16>);

REGISTER_OP_MLU_KERNEL(hard_swish,
                       ops::HardSwishMLUKernel<float>,
                       ops::HardSwishMLUKernel<paddle::platform::float16>);

REGISTER_OP_MLU_KERNEL(hard_swish_grad,
                       ops::HardSwishGradMLUKernel<float>,
                       ops::HardSwishGradMLUKernel<paddle::platform::float16>);

REGISTER_OP_MLU_KERNEL(hard_sigmoid,
                       ops::HardSigmoidMLUKernel<float>,
                       ops::HardSigmoidMLUKernel<paddle::platform::float16>);

REGISTER_OP_MLU_KERNEL(
    hard_sigmoid_grad,
    ops::HardSigmoidGradMLUKernel<float>,
    ops::HardSigmoidGradMLUKernel<paddle::platform::float16>);

REGISTER_OP_MLU_KERNEL(floor,
                       ops::FloorMLUKernel<float>,
                       ops::FloorMLUKernel<paddle::platform::float16>);
