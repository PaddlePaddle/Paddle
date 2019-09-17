//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename InT>
struct OneHotV2OpFunctor {
  const framework::LoDTensor* in_;
  framework::LoDTensor* out_;
  int depth_;
  const DeviceContext& ctx_;
  bool allow_out_of_range_;

  OneHotV2OpFunctor(const framework::LoDTensor* in, framework::LoDTensor* out,
                    int depth, const DeviceContext& ctx,
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
    math::set_constant(ctx_, out_, 0.0);

    if (allow_out_of_range_) {
      for (int i = 0; i < numel; ++i) {
        if (p_in_data[i] >= 0 && p_in_data[i] < depth_) {
          *(p_out_data + i * depth_ + p_in_data[i]) = 1.0;
        }
      }
    } else {
      for (int i = 0; i < numel; ++i) {
        PADDLE_ENFORCE_GE(p_in_data[i], 0,
                          "Illegal index value, should be at least 0.");
        PADDLE_ENFORCE_LT(
            p_in_data[i], depth_,
            "Illegal index value, should be less than depth (%d).", depth_);
        *(p_out_data + i * depth_ + p_in_data[i]) = 1.0;
      }
    }
  }
};

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;
template <typename DeviceContext, typename T>
class OneHotV2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    int depth = context.Attr<int>("depth");
    bool allow_out_of_range = context.Attr<bool>("allow_out_of_range");
    if (context.HasInput("depth_tensor")) {
      auto* depth_tensor = context.Input<Tensor>("depth_tensor");
      auto* depth_data = depth_tensor->data<int32_t>();
      depth = depth_data[0];
      auto out_dims = out->dims();
      out_dims[out_dims.size() - 1] = depth;
      out->Resize(out_dims);
    }

    framework::VisitDataType(
        static_cast<framework::proto::VarType::Type>(
            context.Attr<int>("dtype")),
        OneHotV2OpFunctor<DeviceContext, T>(
            in, out, depth, context.template device_context<DeviceContext>(),
            allow_out_of_range));
  }
};

}  // namespace operators
}  // namespace paddle
