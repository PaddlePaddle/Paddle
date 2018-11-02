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

#include <algorithm>
#include "paddle/fluid/operators/mask_extract_op.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using platform::PADDLE_CUDA_NUM_THREADS;

__global__ void MaskBinarize(const int64_t* mask_data, int64_t n,
                             int64_t* offset) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    offset[idx] = (mask_data[idx] >= 0);
  } else if (idx == n) {
    offset[idx] = 0;
  }
}

template <typename T>
__global__ void MaskExtract(const T* x_data, const int64_t* mask_data,
                            const int64_t* offset_data, int64_t n,
                            int64_t feat_dim, T* out_data, int64_t* ids_data) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n && (offset_data[idx] < offset_data[idx + 1])) {
    ids_data[offset_data[idx]] = mask_data[idx];
    int64_t i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < feat_dim) {
      out_data[offset_data[idx] * feat_dim + i] = x_data[idx * feat_dim + i];
    }
  }
}

template <typename T>
__global__ void MaskExtractGrad(const T* d_out_data, const int64_t* offset_data,
                                int64_t feat_dim, int64_t n, T* d_x_data) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx < n && idy < feat_dim) {
    if (offset_data[idx] < offset_data[idx + 1]) {
      d_x_data[idx * feat_dim + idy] =
          d_out_data[offset_data[idx] * feat_dim + idy];
    } else {
      d_x_data[idx * feat_dim + idy] = static_cast<T>(0);
    }
  }
}

template <typename DeviceContext, typename T>
class MaskExtractGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* mask = ctx.Input<framework::LoDTensor>("Mask");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    auto* ids = ctx.Output<framework::LoDTensor>("Ids");
    auto* offset = ctx.Output<framework::LoDTensor>("Offset");

    auto x_dims = x->dims();
    auto out_dims = x_dims;

    offset->Resize({x_dims[0] + 1, 1});
    offset->mutable_data<int64_t>(ctx.GetPlace());

    auto stream = ctx.cuda_device_context().stream();
    MaskBinarize<<<x_dims[0] / PADDLE_CUDA_NUM_THREADS + 1,
                   PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        mask->data<int64_t>(), x_dims[0], offset->data<int64_t>());

    thrust::device_ptr<int64_t> offset_ptr =
        thrust::device_pointer_cast(offset->data<int64_t>());
    thrust::exclusive_scan(offset_ptr, offset_ptr + x_dims[0] + 1, offset_ptr);

    int64_t out_len;
    platform::CUDAPlace place = boost::get<platform::CUDAPlace>(ctx.GetPlace());
    memory::Copy(platform::CPUPlace(), &out_len, place,
                 offset->data<int64_t>() + x_dims[0], sizeof(int64_t), stream);

    ids->Resize({out_len, 1});
    ids->mutable_data<int64_t>(ctx.GetPlace());
    out_dims[0] = out_len;
    out->Resize(out_dims);
    out->mutable_data<T>(ctx.GetPlace());

    auto feat_dim = x->numel() / x_dims[0];
    int64_t kBlockSize = PADDLE_CUDA_NUM_THREADS;
    size_t y_threads =
        std::min(((((feat_dim + 7) >> 3) + 31) >> 5) << 5, kBlockSize);
    size_t x_threads = (kBlockSize - 1) / y_threads + 1;

    dim3 threads(x_threads, y_threads);
    dim3 blocks((x_dims[0] - 1) / x_threads + 1,
                (feat_dim - 1) / y_threads + 1);

    MaskExtract<<<blocks, threads, 0, stream>>>(
        x->data<T>(), mask->data<int64_t>(), offset->data<int64_t>(), x_dims[0],
        feat_dim, out->data<T>(), ids->data<int64_t>());
  }
};

template <typename DeviceContext, typename T>
class MaskExtractGPUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* d_out =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto* offset = ctx.Input<framework::LoDTensor>("Offset");
    auto* d_x = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));

    d_x->mutable_data<T>(ctx.GetPlace());
    auto x_dims = d_x->dims();
    auto feat_dim = d_x->numel() / x_dims[0];

    int64_t kBlockSize = PADDLE_CUDA_NUM_THREADS;
    size_t y_threads =
        std::min(((((feat_dim + 7) >> 3) + 31) >> 5) << 5, kBlockSize);
    size_t x_threads = (kBlockSize - 1) / y_threads + 1;
    dim3 threads(x_threads, y_threads);
    dim3 blocks((x_dims[0] - 1) / x_threads + 1,
                (feat_dim - 1) / y_threads + 1);

    auto stream = ctx.cuda_device_context().stream();
    MaskExtractGrad<<<blocks, threads, 0, stream>>>(
        d_out->data<T>(), offset->data<int64_t>(), feat_dim, x_dims[0],
        d_x->data<T>());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    mask_extract,
    ops::MaskExtractGPUKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MaskExtractGPUKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MaskExtractGPUKernel<paddle::platform::CUDADeviceContext, int>,
    ops::MaskExtractGPUKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    mask_extract_grad,
    ops::MaskExtractGPUGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MaskExtractGPUGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MaskExtractGPUGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::MaskExtractGPUGradKernel<paddle::platform::CUDADeviceContext,
                                  int64_t>);
