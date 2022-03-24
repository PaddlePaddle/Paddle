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

template <typename DeviceContext, typename T>
class EqualMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<bool>(ctx.GetPlace());

    MLUCnnlTensorDesc input_x(*x, CNNL_LAYOUT_ARRAY,
                              ToCnnlDataType(x->dtype()));
    MLUCnnlTensorDesc input_y(*y, CNNL_LAYOUT_ARRAY,
                              ToCnnlDataType(y->dtype()));
    MLUCnnlTensorDesc output(*out, CNNL_LAYOUT_ARRAY,
                             ToCnnlDataType(out->dtype()));
    MLUCnnl::Logic(ctx, CNNL_LOGIC_OP_EQ, input_x.get(), GetBasePtr(x),
                   input_y.get(), GetBasePtr(y), output.get(), GetBasePtr(out));
  }
};

template <typename DeviceContext, typename T>
class NotEqualMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<bool>(ctx.GetPlace());

    MLUCnnlTensorDesc input_x(*x, CNNL_LAYOUT_ARRAY,
                              ToCnnlDataType(x->dtype()));
    MLUCnnlTensorDesc input_y(*y, CNNL_LAYOUT_ARRAY,
                              ToCnnlDataType(y->dtype()));
    MLUCnnlTensorDesc output(*out, CNNL_LAYOUT_ARRAY,
                             ToCnnlDataType(out->dtype()));
    MLUCnnl::Logic(ctx, CNNL_LOGIC_OP_NE, input_x.get(), GetBasePtr(x),
                   input_y.get(), GetBasePtr(y), output.get(), GetBasePtr(out));
  }
};

template <typename DeviceContext, typename T>
class LessThanMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<bool>(ctx.GetPlace());

    MLUCnnlTensorDesc input_x(*x, CNNL_LAYOUT_ARRAY,
                              ToCnnlDataType(x->dtype()));
    MLUCnnlTensorDesc input_y(*y, CNNL_LAYOUT_ARRAY,
                              ToCnnlDataType(y->dtype()));
    MLUCnnlTensorDesc output(*out, CNNL_LAYOUT_ARRAY,
                             ToCnnlDataType(out->dtype()));
    MLUCnnl::Logic(ctx, CNNL_LOGIC_OP_LT, input_x.get(), GetBasePtr(x),
                   input_y.get(), GetBasePtr(y), output.get(), GetBasePtr(out));
  }
};

template <typename DeviceContext, typename T>
class LessEqualMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<bool>(ctx.GetPlace());

    MLUCnnlTensorDesc input_x(*x, CNNL_LAYOUT_ARRAY,
                              ToCnnlDataType(x->dtype()));
    MLUCnnlTensorDesc input_y(*y, CNNL_LAYOUT_ARRAY,
                              ToCnnlDataType(y->dtype()));
    MLUCnnlTensorDesc output(*out, CNNL_LAYOUT_ARRAY,
                             ToCnnlDataType(out->dtype()));
    MLUCnnl::Logic(ctx, CNNL_LOGIC_OP_LE, input_x.get(), GetBasePtr(x),
                   input_y.get(), GetBasePtr(y), output.get(), GetBasePtr(out));
  }
};

template <typename DeviceContext, typename T>
class GreaterThanMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<bool>(ctx.GetPlace());

    MLUCnnlTensorDesc input_x(*x, CNNL_LAYOUT_ARRAY,
                              ToCnnlDataType(x->dtype()));
    MLUCnnlTensorDesc input_y(*y, CNNL_LAYOUT_ARRAY,
                              ToCnnlDataType(y->dtype()));
    MLUCnnlTensorDesc output(*out, CNNL_LAYOUT_ARRAY,
                             ToCnnlDataType(out->dtype()));
    MLUCnnl::Logic(ctx, CNNL_LOGIC_OP_GT, input_x.get(), GetBasePtr(x),
                   input_y.get(), GetBasePtr(y), output.get(), GetBasePtr(out));
  }
};

template <typename DeviceContext, typename T>
class GreaterEqualMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<bool>(ctx.GetPlace());

    MLUCnnlTensorDesc input_x(*x, CNNL_LAYOUT_ARRAY,
                              ToCnnlDataType(x->dtype()));
    MLUCnnlTensorDesc input_y(*y, CNNL_LAYOUT_ARRAY,
                              ToCnnlDataType(y->dtype()));
    MLUCnnlTensorDesc output(*out, CNNL_LAYOUT_ARRAY,
                             ToCnnlDataType(out->dtype()));
    MLUCnnl::Logic(ctx, CNNL_LOGIC_OP_GE, input_x.get(), GetBasePtr(x),
                   input_y.get(), GetBasePtr(y), output.get(), GetBasePtr(out));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(
    equal, ops::EqualMLUKernel<plat::MLUDeviceContext, plat::float16>,
    ops::EqualMLUKernel<plat::MLUDeviceContext, float>,
    ops::EqualMLUKernel<plat::MLUDeviceContext, int8_t>,
    ops::EqualMLUKernel<plat::MLUDeviceContext, uint8_t>,
    ops::EqualMLUKernel<plat::MLUDeviceContext, int16_t>,
    ops::EqualMLUKernel<plat::MLUDeviceContext, int>,
    ops::EqualMLUKernel<plat::MLUDeviceContext, bool>);

REGISTER_OP_MLU_KERNEL(
    not_equal, ops::NotEqualMLUKernel<plat::MLUDeviceContext, plat::float16>,
    ops::NotEqualMLUKernel<plat::MLUDeviceContext, float>,
    ops::NotEqualMLUKernel<plat::MLUDeviceContext, int8_t>,
    ops::NotEqualMLUKernel<plat::MLUDeviceContext, uint8_t>,
    ops::NotEqualMLUKernel<plat::MLUDeviceContext, int16_t>,
    ops::NotEqualMLUKernel<plat::MLUDeviceContext, int>,
    ops::NotEqualMLUKernel<plat::MLUDeviceContext, bool>);

REGISTER_OP_MLU_KERNEL(
    less_than, ops::LessThanMLUKernel<plat::MLUDeviceContext, plat::float16>,
    ops::LessThanMLUKernel<plat::MLUDeviceContext, float>,
    ops::LessThanMLUKernel<plat::MLUDeviceContext, int8_t>,
    ops::LessThanMLUKernel<plat::MLUDeviceContext, uint8_t>,
    ops::LessThanMLUKernel<plat::MLUDeviceContext, int16_t>,
    ops::LessThanMLUKernel<plat::MLUDeviceContext, int>,
    ops::LessThanMLUKernel<plat::MLUDeviceContext, bool>);

REGISTER_OP_MLU_KERNEL(
    less_equal, ops::LessEqualMLUKernel<plat::MLUDeviceContext, plat::float16>,
    ops::LessEqualMLUKernel<plat::MLUDeviceContext, float>,
    ops::LessEqualMLUKernel<plat::MLUDeviceContext, int8_t>,
    ops::LessEqualMLUKernel<plat::MLUDeviceContext, uint8_t>,
    ops::LessEqualMLUKernel<plat::MLUDeviceContext, int16_t>,
    ops::LessEqualMLUKernel<plat::MLUDeviceContext, int>,
    ops::LessEqualMLUKernel<plat::MLUDeviceContext, bool>);

REGISTER_OP_MLU_KERNEL(
    greater_than,
    ops::GreaterThanMLUKernel<plat::MLUDeviceContext, plat::float16>,
    ops::GreaterThanMLUKernel<plat::MLUDeviceContext, float>,
    ops::GreaterThanMLUKernel<plat::MLUDeviceContext, int8_t>,
    ops::GreaterThanMLUKernel<plat::MLUDeviceContext, uint8_t>,
    ops::GreaterThanMLUKernel<plat::MLUDeviceContext, int16_t>,
    ops::GreaterThanMLUKernel<plat::MLUDeviceContext, int>,
    ops::GreaterThanMLUKernel<plat::MLUDeviceContext, bool>);

REGISTER_OP_MLU_KERNEL(
    greater_equal,
    ops::GreaterEqualMLUKernel<plat::MLUDeviceContext, plat::float16>,
    ops::GreaterEqualMLUKernel<plat::MLUDeviceContext, float>,
    ops::GreaterEqualMLUKernel<plat::MLUDeviceContext, int8_t>,
    ops::GreaterEqualMLUKernel<plat::MLUDeviceContext, uint8_t>,
    ops::GreaterEqualMLUKernel<plat::MLUDeviceContext, int16_t>,
    ops::GreaterEqualMLUKernel<plat::MLUDeviceContext, int>,
    ops::GreaterEqualMLUKernel<plat::MLUDeviceContext, bool>);
