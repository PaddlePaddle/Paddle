// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/kernels/one_hot_kernel.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/pten/core/kernel_registry.h"

namespace pten {

template <typename DeviceContext, typename InT>
struct OneHotV2OpFunctor {
  const DenseTensor* in_;
  DenseTensor* out_;
  int depth_;
  const DeviceContext& ctx_;
  bool allow_out_of_range_;

  OneHotV2OpFunctor(const DenseTensor* in,
                    DenseTensor* out,
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
    paddle::operators::math::set_constant(ctx_, out_, 0.0);

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

template <typename T, typename Context>
void OneHotKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& depth,
                  int dtype,
                  bool allow_out_of_range,
                  DenseTensor* out) {
  int depth_val = depth.to<int>();
  auto out_dims = out->dims();
  if (out_dims[out_dims.size() - 1] == -1) {
    out_dims[out_dims.size() - 1] = depth_val;
    out->Resize(out_dims);
  }
  out->mutable_data<T>(dev_ctx.GetPlace());
  paddle::framework::VisitDataType(
      static_cast<paddle::framework::proto::VarType::Type>(dtype),
      OneHotV2OpFunctor<Context, T>(
          &x, out, depth_val, dev_ctx, allow_out_of_range));
}

}  // namespace pten

PT_REGISTER_KERNEL(
    one_hot_v2, CPU, ALL_LAYOUT, pten::OneHotKernel, int, int64_t) {}
