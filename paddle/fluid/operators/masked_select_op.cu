/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reverse.h>
#include <thrust/scan.h>
#include "paddle/fluid/operators/masked_select_op.h"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DDim = framework::DDim;

__global__ void SetMaskArray(const bool* mask, int32_t* mask_array, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < size; idx += blockDim.x * gridDim.x) {
    if (mask[idx])
      mask_array[idx] = 1;
    else
      mask_array[idx] = 0;
  }
}

template <typename T>
__global__ void SelectWithPrefixMask(const int32_t* mask_prefix_sum,
                                     const bool* mask, const T* input, T* out,
                                     int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < size; idx += blockDim.x * gridDim.x) {
    if (mask[idx]) {
      int index = mask_prefix_sum[idx];
      out[index] = input[idx];
    }
  }
}

template <typename T>
__global__ void SelectGradWithPrefixMask(const int32_t* mask_prefix_sum,
                                         const bool* mask, const T* input,
                                         T* out, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < size; idx += blockDim.x * gridDim.x) {
    if (mask[idx]) {
      int index = mask_prefix_sum[idx];
      out[idx] = input[index];
    } else {
      out[idx] = 0;
    }
  }
}

template <typename DeviceContext, typename T>
class MaskedSelectCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto input = ctx.Input<framework::Tensor>("X");
    auto mask = ctx.Input<framework::Tensor>("Mask");
    auto out = ctx.Output<framework::Tensor>("Y");
    auto* mask_data = mask->data<bool>();
    auto input_data = input->data<T>();

    auto mask_size = mask->numel();
    auto input_dim = input->dims();
    auto mask_dim = mask->dims();
    PADDLE_ENFORCE_EQ(
        input_dim, mask_dim,
        platform::errors::InvalidArgument(
            "The dim size of input and mask in OP(masked_selected) "
            "must be equal, but got input dim:(%ld), mask dim: "
            "(%ld). Please check input "
            "value.",
            input_dim, mask_dim));

    thrust::device_ptr<const bool> mask_dev_ptr =
        thrust::device_pointer_cast(mask_data);
    thrust::device_vector<T> mask_vec(mask_dev_ptr, mask_dev_ptr + mask_size);
    auto out_size = thrust::count(mask_vec.begin(), mask_vec.end(), true);

    framework::DDim out_dim{out_size};
    out->Resize(out_dim);
    auto out_data = out->mutable_data<T>(ctx.GetPlace());

    Tensor mask_array;
    Tensor mask_prefix_sum;
    mask_array.Resize(mask_dim);
    mask_prefix_sum.Resize(mask_dim);

    int32_t* mask_array_data = mask_array.mutable_data<int32_t>(ctx.GetPlace());
    int32_t* mask_prefix_sum_data =
        mask_prefix_sum.mutable_data<int32_t>(ctx.GetPlace());
    int threads = 512;
    int grid = (mask_size + threads - 1) / threads;
    auto stream = ctx.cuda_device_context().stream();
    SetMaskArray<<<grid, threads, 0, stream>>>(mask_data, mask_array_data,
                                               mask_size);

    thrust::device_ptr<int32_t> mask_array_dev_ptr =
        thrust::device_pointer_cast(mask_array_data);
    thrust::device_vector<int32_t> mask_array_vec(
        mask_array_dev_ptr, mask_array_dev_ptr + mask_size);
    thrust::exclusive_scan(thrust::device, mask_array_vec.begin(),
                           mask_array_vec.end(), mask_prefix_sum_data);

    SelectWithPrefixMask<T><<<grid, threads, 0, stream>>>(
        mask_prefix_sum_data, mask_data, input_data, out_data, mask_size);
  }
};

template <typename DeviceContext, typename T>
class MaskedSelectGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto input = ctx.Input<framework::Tensor>(framework::GradVarName("Y"));
    auto mask = ctx.Input<framework::Tensor>("Mask");
    auto out = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* mask_data = mask->data<bool>();
    auto* input_data = input->data<T>();
    auto* out_data = out->mutable_data<T>(ctx.GetPlace());

    auto input_size = input->numel();
    auto mask_size = mask->numel();
    auto mask_dim = mask->dims();

    auto out_size = mask_size;

    Tensor mask_array;
    Tensor mask_prefix_sum;
    mask_array.Resize(mask_dim);
    mask_prefix_sum.Resize(mask_dim);

    int32_t* mask_array_data = mask_array.mutable_data<int32_t>(ctx.GetPlace());
    int32_t* mask_prefix_sum_data =
        mask_prefix_sum.mutable_data<int32_t>(ctx.GetPlace());
    int threads = 512;
    int grid = (mask_size + threads - 1) / threads;
    auto stream = ctx.cuda_device_context().stream();
    SetMaskArray<<<grid, threads, 0, stream>>>(mask_data, mask_array_data,
                                               mask_size);

    thrust::device_ptr<int32_t> mask_array_dev_ptr =
        thrust::device_pointer_cast(mask_array_data);
    thrust::device_vector<int32_t> mask_array_vec(
        mask_array_dev_ptr, mask_array_dev_ptr + mask_size);
    thrust::exclusive_scan(thrust::device, mask_array_vec.begin(),
                           mask_array_vec.end(), mask_prefix_sum_data);

    SelectGradWithPrefixMask<T><<<grid, threads, 0, stream>>>(
        mask_prefix_sum_data, mask_data, input_data, out_data, mask_size);
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    masked_select,
    ops::MaskedSelectCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MaskedSelectCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MaskedSelectCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::MaskedSelectCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    masked_select_grad,
    ops::MaskedSelectGradCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MaskedSelectGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                    double>,
    ops::MaskedSelectGradCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::MaskedSelectGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                    int64_t>);
