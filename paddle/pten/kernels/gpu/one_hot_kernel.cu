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

#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/one_hot_kernel.h"

namespace pten {

using paddle::platform::PADDLE_CUDA_NUM_THREADS;

template <typename InT, typename OutT>
__global__ void FillOutputKernel(const InT* p_in_data,
                                 OutT* p_out_data,
                                 const int64_t numel,
                                 const int depth) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel && p_in_data[idx] >= 0 && p_in_data[idx] < depth) {
    *(p_out_data + (idx * depth) + p_in_data[idx]) = 1.0;
  }
}

template <typename DeviceContext, typename InT>
struct OneHotV2OpCUDAFunctor {
  const DenseTensor* in_;
  DenseTensor* out_;
  const DeviceContext& ctx_;
  int depth_;

  OneHotV2OpCUDAFunctor(const DenseTensor* in,
                        DenseTensor* out,
                        int depth,
                        const DeviceContext& ctx)
      : in_(in), out_(out), depth_(depth), ctx_(ctx) {}

  template <typename OutT>
  void apply() const {
    auto* p_in_data = in_->data<InT>();
    auto numel = in_->numel();
    auto* p_out_data = out_->mutable_data<OutT>(ctx_.GetPlace());
    auto stream = ctx_.stream();
    paddle::operators::math::set_constant(ctx_, out_, 0.0);

    FillOutputKernel<<<(numel + PADDLE_CUDA_NUM_THREADS - 1) /
                           PADDLE_CUDA_NUM_THREADS,
                       PADDLE_CUDA_NUM_THREADS,
                       0,
                       stream>>>(p_in_data, p_out_data, numel, depth_);
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
      OneHotV2OpCUDAFunctor<Context, T>(&x, out, depth_val, dev_ctx));
}

}  // namespace pten

PT_REGISTER_KERNEL(one_hot, GPU, ALL_LAYOUT, pten::OneHotKernel, int, int64_t) {
}
