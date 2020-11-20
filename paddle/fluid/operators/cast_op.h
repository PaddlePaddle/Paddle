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

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

template <typename InT, typename OutT>
struct CastOpTransformFunctor {
  HOSTDEVICE OutT operator()(InT in) const { return static_cast<OutT>(in); }
};

template <typename DeviceContext, typename InT>
struct CastOpFunctor {
  const framework::Tensor* in_;
  framework::Tensor* out_;
  const DeviceContext& ctx_;
  CastOpFunctor(const framework::Tensor* in, framework::Tensor* out,
                const DeviceContext& ctx)
      : in_(in), out_(out), ctx_(ctx) {}

  template <typename OutT>
  void apply() const {
    auto* in_begin = in_->data<InT>();
    auto numel = in_->numel();
    auto* in_end = in_begin + numel;
    auto* out_begin = out_->mutable_data<OutT>(ctx_.GetPlace());
    platform::Transform<DeviceContext> trans;
    trans(ctx_, in_begin, in_end, out_begin,
          CastOpTransformFunctor<InT, OutT>());
  }
};

template <typename DeviceContext, typename InT, typename OutT>
static void CastFunction(const framework::ExecutionContext& context) {
  auto* in = context.Input<framework::Tensor>("X");
  auto* out = context.Output<framework::Tensor>("Out");

  auto in_t = framework::EigenVector<InT>::Flatten(*in);
  out->mutable_data<OutT>(context.GetPlace());
  auto out_t = framework::EigenVector<OutT>::Flatten(*out);
  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();
  out_t.device(place) = in_t.template cast<OutT>();
}

template <typename DeviceContext, typename InT>
class CastOpKernel : public framework::OpKernel<InT> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto out_type = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("out_dtype"));

    if (out_type == paddle::framework::proto::VarType::FP64) {
      CastFunction<DeviceContext, InT, double>(context);
    } else if (out_type == paddle::framework::proto::VarType::FP32) {
      CastFunction<DeviceContext, InT, float>(context);
    } else if (out_type == paddle::framework::proto::VarType::FP16) {
      CastFunction<DeviceContext, InT, paddle::platform::float16>(context);
    } else if (out_type == paddle::framework::proto::VarType::INT64) {
      CastFunction<DeviceContext, InT, int64_t>(context);
    } else if (out_type == paddle::framework::proto::VarType::INT32) {
      CastFunction<DeviceContext, InT, int>(context);
    } else if (out_type == paddle::framework::proto::VarType::UINT8) {
      CastFunction<DeviceContext, InT, uint8_t>(context);
    } else if (out_type == paddle::framework::proto::VarType::BOOL) {
      CastFunction<DeviceContext, InT, bool>(context);
    }
  }
};

}  // namespace operators
}  // namespace paddle
