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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/transform.h"

#include "paddle/fluid/framework/pten_utils.h"
#include "paddle/pten/api/lib/device_context_pool.h"
#include "paddle/pten/api/lib/utils/tensor_utils.h"
#include "paddle/pten/core/context.h"
#include "paddle/pten/include/core.h"
#include "paddle/pten/include/manipulation.h"

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

template <typename DeviceContext, typename InT>
class CastOpKernel : public framework::OpKernel<InT> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");

    auto out_dtype = context.Attr<int>("out_dtype");
    // todo: not used in_dtype
    auto in_dtype = context.Attr<int>("in_dtype");

    out->mutable_data(context.GetPlace(),
                      static_cast<framework::proto::VarType::Type>(out_dtype));

    auto pt_x = paddle::experimental::MakePtenDenseTensor(*in);
    auto pt_out = paddle::experimental::MakePtenDenseTensor(*out);

    auto pt_out_dtype = pten::TransToPtenDataType(
        static_cast<framework::proto::VarType::Type>(out_dtype));
    auto pt_in_dtype = pten::TransToPtenDataType(
        static_cast<framework::proto::VarType::Type>(in_dtype));

    // call new kernel
    auto* dev_ctx = reinterpret_cast<
        typename framework::ConvertContextType<DeviceContext>::TYPE*>(
        paddle::experimental::DeviceContextPool::Instance().Get(
            context.GetPlace()));
    pten::Cast<InT>(*dev_ctx, *pt_x.get(), pt_out_dtype, pt_in_dtype,
                    pt_out.get());
  }
};

}  // namespace operators
}  // namespace paddle
