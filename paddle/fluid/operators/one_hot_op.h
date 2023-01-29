//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename InT>
struct OneHotOpFunctor {
  const phi::DenseTensor* in_;
  phi::DenseTensor* out_;
  int depth_;
  const DeviceContext& ctx_;
  bool allow_out_of_range_;

  OneHotOpFunctor(const phi::DenseTensor* in,
                  phi::DenseTensor* out,
                  int depth,
                  const DeviceContext& ctx,
                  bool allow_out_of_range = false)
      : in_(in),
        out_(out),
        depth_(depth),
        ctx_(ctx),
        allow_out_of_range_(allow_out_of_range) {}

  template <typename OutT>
  void apply() const {
    auto* p_in_data = in_->data<InT>();
    auto numel = in_->numel();
    auto* p_out_data = out_->mutable_data<OutT>(ctx_.GetPlace());
    phi::funcs::set_constant(ctx_, out_, 0.0);

    if (allow_out_of_range_) {
      for (int i = 0; i < numel; ++i) {
        if (p_in_data[i] >= 0 && p_in_data[i] < depth_) {
          *(p_out_data + i * depth_ + p_in_data[i]) = 1.0;
        }
      }
    } else {
      for (int i = 0; i < numel; ++i) {
        PADDLE_ENFORCE_GE(
            p_in_data[i],
            0,
            platform::errors::InvalidArgument(
                "Illegal index value, Input(input) value should be at least 0, "
                "but received input (%d) less than 0",
                p_in_data[i]));
        PADDLE_ENFORCE_LT(
            p_in_data[i],
            depth_,
            platform::errors::InvalidArgument(
                "Illegal index value, Input(input) value should be less than "
                "Input(depth), "
                "but received input (%d) not less than depth (%d)",
                p_in_data[i],
                depth_));

        *(p_out_data + i * depth_ + p_in_data[i]) = 1.0;
      }
    }
  }
};

template <typename DeviceContext, typename T>
class OneHotKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<phi::DenseTensor>("X");
    auto* out = context.Output<phi::DenseTensor>("Out");
    int depth = context.Attr<int>("depth");
    bool allow_out_of_range = context.Attr<bool>("allow_out_of_range");
    if (context.HasInput("depth_tensor")) {
      auto* depth_tensor = context.Input<phi::DenseTensor>("depth_tensor");
      auto* depth_data = depth_tensor->data<int32_t>();
      depth = depth_data[0];
      auto in_dims = in->dims();
      framework::DDim out_dims(in_dims);
      out_dims[out_dims.size() - 1] = depth;
      out->Resize(out_dims);
    }

    framework::VisitDataType(
        static_cast<framework::proto::VarType::Type>(
            context.Attr<int>("dtype")),
        OneHotOpFunctor<DeviceContext, T>(
            in,
            out,
            depth,
            context.template device_context<DeviceContext>(),
            allow_out_of_range));
  }
};

}  // namespace operators
}  // namespace paddle
