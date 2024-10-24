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

#include "paddle/phi/kernels/one_hot_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename DeviceContext, typename InT>
struct OneHotV2OpFunctor {
  const DenseTensor* in_;
  DenseTensor* out_;
  int depth_;
  const DeviceContext& ctx_;

  OneHotV2OpFunctor(const DenseTensor* in,
                    DenseTensor* out,
                    int depth,
                    const DeviceContext& ctx)
      : in_(in), out_(out), depth_(depth), ctx_(ctx) {}

  template <typename OutT>
  void apply() const {
    auto* p_in_data = in_->data<InT>();
    auto numel = in_->numel();
    auto* p_out_data = ctx_.template Alloc<OutT>(out_);
    funcs::set_constant(ctx_, out_, static_cast<OutT>(0.0));

    for (int i = 0; i < numel; ++i) {
      PADDLE_ENFORCE_GE(
          p_in_data[i],
          0,
          common::errors::InvalidArgument(
              "Illegal index value, Input(input) value should be at least 0, "
              "but received input (%d) less than 0",
              p_in_data[i]));
      PADDLE_ENFORCE_LT(
          p_in_data[i],
          depth_,
          common::errors::InvalidArgument(
              "Illegal index value, Input(input) value should be less than "
              "Input(depth), "
              "but received input (%d) not less than depth (%d)",
              p_in_data[i],
              depth_));
      *(p_out_data + i * depth_ + p_in_data[i]) = 1.0;
    }
  }
};

template <typename T, typename Context>
void OneHotKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& depth,
                  DenseTensor* out) {
  auto depth_v = depth.to<int>();
  auto out_dims = out->dims();
  if (out_dims[out_dims.size() - 1] == -1) {
    out_dims[out_dims.size() - 1] = depth_v;
    out->Resize(out_dims);
  }

  auto* p_in_data = x.data<T>();
  auto numel = x.numel();
  auto* p_out_data = dev_ctx.template Alloc<float>(out);
  funcs::set_constant(dev_ctx, out, 0.0f);

  for (int i = 0; i < numel; ++i) {
    PADDLE_ENFORCE_GE(
        p_in_data[i],
        0,
        common::errors::InvalidArgument(
            "Illegal index value, Input(input) value should be at least 0, "
            "but received input (%d) less than 0",
            p_in_data[i]));
    PADDLE_ENFORCE_LT(
        p_in_data[i],
        depth_v,
        common::errors::InvalidArgument(
            "Illegal index value, Input(input) value should be less than "
            "Input(depth), "
            "but received input (%d) not less than depth (%d)",
            p_in_data[i],
            depth_v));
    *(p_out_data + i * depth_v + p_in_data[i]) = 1.0;
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(one_hot, CPU, ALL_LAYOUT, phi::OneHotKernel, int, int64_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
}
