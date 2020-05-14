/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <thrust/device_vector.h>
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/slice_op.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

template <size_t D>
__global__ void Padding(const paddle::platform::float16* d_out,
                        const int64_t* out_dims, const int64_t* in_dims,
                        const int64_t* offsets, int64_t n,
                        paddle::platform::float16* d_in) {
  int64_t out_idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (out_idx < n) {
    int64_t out_idx_tmp = out_idx;
    int64_t coords[D] = {0};
    for (int i = D - 1; i >= 0; --i) {
      coords[i] = out_idx_tmp % out_dims[i];
      out_idx_tmp /= out_dims[i];
      coords[i] += offsets[i];
    }

    int64_t in_idx = 0;
    for (int i = 0; i < D; ++i) {
      in_idx = in_idx * in_dims[i] + coords[i];
    }

    d_in[in_idx] = d_out[out_idx];
  }
}

template <>
class SliceGradKernel<paddle::platform::CUDADeviceContext,
                      paddle::platform::float16>
    : public framework::OpKernel<paddle::platform::float16> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* d_out = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* d_in = ctx.Output<framework::Tensor>(framework::GradVarName("Input"));
    d_in->mutable_data<paddle::platform::float16>(ctx.GetPlace());

    auto out_dims = d_out->dims();
    auto in_dims = d_in->dims();
    int rank = out_dims.size();
    std::vector<int64_t> offsets(rank, 0);
    auto axes = ctx.Attr<std::vector<int>>("axes");
    auto starts_int = ctx.Attr<std::vector<int>>("starts");
    std::vector<int64_t> starts(starts_int.begin(), starts_int.end());

    auto list_new_starts_tensor =
        ctx.MultiInput<framework::Tensor>("StartsTensorList");

    if (list_new_starts_tensor.size() > 0) {
      starts = GetDataFromTensorList<int64_t>(list_new_starts_tensor);
    } else if (ctx.HasInput("StartsTensor")) {
      auto* starts_tensor = ctx.Input<framework::Tensor>("StartsTensor");
      starts = GetDataFromTensor<int64_t>(starts_tensor);
    }

    for (size_t i = 0; i < starts.size(); ++i) {
      if (starts[i] < 0) {
        starts[i] += in_dims[axes[i]];
      }
      offsets[axes[i]] = std::max(starts[i], static_cast<int64_t>(0));
    }

    math::SetConstant<paddle::platform::CUDADeviceContext,
                      paddle::platform::float16>
        set_zero;
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::CUDADeviceContext>();
    set_zero(dev_ctx, d_in, static_cast<paddle::platform::float16>(0));

    int64_t numel = d_out->numel();
    dim3 blocks((numel - 1) / PADDLE_CUDA_NUM_THREADS + 1);
    dim3 threads(PADDLE_CUDA_NUM_THREADS);
    auto stream = ctx.cuda_device_context().stream();

    auto out_shape = framework::vectorize<int64_t>(out_dims);
    thrust::device_vector<int64_t> out_dims_vec(out_shape.begin(),
                                                out_shape.end());
    auto in_shape = framework::vectorize<int64_t>(in_dims);
    thrust::device_vector<int64_t> in_dims_vec(in_shape.begin(),
                                               in_shape.end());
    thrust::device_vector<int64_t> offsets_vec(offsets.begin(), offsets.end());
    const int64_t* out_dims_ptr = thrust::raw_pointer_cast(out_dims_vec.data());
    const int64_t* in_dims_ptr = thrust::raw_pointer_cast(in_dims_vec.data());
    const int64_t* offsets_ptr = thrust::raw_pointer_cast(offsets_vec.data());

    switch (rank) {
      case 1:
        Padding<1><<<blocks, threads, 0, stream>>>(
            d_out->data<paddle::platform::float16>(), out_dims_ptr, in_dims_ptr,
            offsets_ptr, numel, d_in->data<paddle::platform::float16>());
        break;
      case 2:
        Padding<2><<<blocks, threads, 0, stream>>>(
            d_out->data<paddle::platform::float16>(), out_dims_ptr, in_dims_ptr,
            offsets_ptr, numel, d_in->data<paddle::platform::float16>());
        break;
      case 3:
        Padding<3><<<blocks, threads, 0, stream>>>(
            d_out->data<paddle::platform::float16>(), out_dims_ptr, in_dims_ptr,
            offsets_ptr, numel, d_in->data<paddle::platform::float16>());
        break;
      case 4:
        Padding<4><<<blocks, threads, 0, stream>>>(
            d_out->data<paddle::platform::float16>(), out_dims_ptr, in_dims_ptr,
            offsets_ptr, numel, d_in->data<paddle::platform::float16>());
        break;
      case 5:
        Padding<5><<<blocks, threads, 0, stream>>>(
            d_out->data<paddle::platform::float16>(), out_dims_ptr, in_dims_ptr,
            offsets_ptr, numel, d_in->data<paddle::platform::float16>());
        break;
      case 6:
        Padding<6><<<blocks, threads, 0, stream>>>(
            d_out->data<paddle::platform::float16>(), out_dims_ptr, in_dims_ptr,
            offsets_ptr, numel, d_in->data<paddle::platform::float16>());
        break;
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    slice, ops::SliceKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, plat::float16>);

REGISTER_OP_CUDA_KERNEL(
    slice_grad,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, plat::float16>);
