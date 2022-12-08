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
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {
template <typename Context, typename InT>
struct OneHotV2OpFunctor {
  const DenseTensor* in_;
  DenseTensor* out_;
  int depth_;
  const Context& ctx_;

  OneHotV2OpFunctor(const DenseTensor* in,
                    DenseTensor* out,
                    int depth,
                    const Context& ctx)
      : in_(in), out_(out), depth_(depth), ctx_(ctx) {}

  template <typename OutT>
  void apply() const {
    auto* p_in_data = in_->data<InT>();
    auto numel = in_->numel();
    auto* p_out_data = ctx_.template Alloc<float>(out_);
    int r = xpu::one_hot<InT>(
        ctx_.x_context(), p_in_data, p_out_data, numel, depth_, 1.0, 0.0);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "one_hot");
  }
};

template <typename T, typename Context>
void OneHotRawKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const Scalar& depth,
                     DataType dtype,
                     bool allow_out_of_range,
                     DenseTensor* out) {
  auto depth_v = depth.to<int>();
  auto out_dims = out->dims();
  if (out_dims[out_dims.size() - 1] == -1) {
    out_dims[out_dims.size() - 1] = depth_v;
    out->Resize(out_dims);
  }
  phi::VisitDataType(dtype,
                     OneHotV2OpFunctor<Context, T>(&x, out, depth_v, dev_ctx));
}
}  // namespace phi

PD_REGISTER_KERNEL(
    one_hot_raw, XPU, ALL_LAYOUT, phi::OneHotRawKernel, int, int64_t) {}
