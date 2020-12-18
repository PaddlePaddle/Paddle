// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/operators/roll_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T>
__global__ void roll_cuda_kernel(const T* input, T* output, int64_t N,
                                 int64_t* shifts, int64_t* strides,
                                 int64_t* sizes, int64_t nums) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }
  int64_t output_idx = idx;
  int64_t dim_idx, dim_idx_shift;
  for (int64_t i = 0; i < nums; i++) {
    dim_idx = idx % (strides[i] * sizes[i]) / strides[i];
    dim_idx_shift = (dim_idx + shifts[i]) % sizes[i];
    output_idx = output_idx + (dim_idx_shift - dim_idx) * strides[i];
  }
  output[output_idx] = input[idx];
}

template <typename DeviceContext, typename T>
class RollCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    std::vector<int64_t> shifts = context.Attr<std::vector<int64_t>>("shifts");
    std::vector<int64_t> dims = context.Attr<std::vector<int64_t>>("axis");

    auto* in_data = in->data<T>();
    auto* out_data = out->mutable_data<T>(context.GetPlace());
    int64_t numel = in->numel();
    auto stream =
        context.template device_context<platform::CUDADeviceContext>().stream();

    size_t nums = shifts.size();
    auto input_dim = in->dims();
    auto stride_dim = framework::stride(input_dim);

    int64_t dim, size;
    size_t gpu_memory_size_ = sizeof(int64_t) * nums;
    std::vector<int64_t> strides, sizes;
    strides.resize(nums);
    sizes.resize(nums);
    paddle::memory::AllocationPtr shifts_gpu =
        memory::Alloc(context.GetPlace(), gpu_memory_size_);
    paddle::memory::AllocationPtr strides_gpu =
        memory::Alloc(context.GetPlace(), gpu_memory_size_);
    paddle::memory::AllocationPtr sizes_gpu =
        memory::Alloc(context.GetPlace(), gpu_memory_size_);

    for (size_t i = 0; i < nums; i++) {
      dim = dims[i] >= 0 ? dims[i] : dims[i] + input_dim.size();
      size = input_dim[dim];
      shifts[i] = (shifts[i] % size + size) % size;
      strides[i] = stride_dim[dim];
      sizes[i] = size;
    }
    paddle::memory::Copy(
        BOOST_GET_CONST(platform::CUDAPlace, shifts_gpu->place()),
        shifts_gpu->ptr(), platform::CPUPlace(), shifts.data(),
        gpu_memory_size_, stream);
    paddle::memory::Copy(
        BOOST_GET_CONST(platform::CUDAPlace, strides_gpu->place()),
        strides_gpu->ptr(), platform::CPUPlace(), strides.data(),
        gpu_memory_size_, stream);
    paddle::memory::Copy(
        BOOST_GET_CONST(platform::CUDAPlace, sizes_gpu->place()),
        sizes_gpu->ptr(), platform::CPUPlace(), sizes.data(), gpu_memory_size_,
        stream);
    int64_t* shifts_ptr = reinterpret_cast<int64_t*>(shifts_gpu->ptr());
    int64_t* strides_ptr = reinterpret_cast<int64_t*>(strides_gpu->ptr());
    int64_t* sizes_ptr = reinterpret_cast<int64_t*>(sizes_gpu->ptr());

    roll_cuda_kernel<<<(numel + PADDLE_CUDA_NUM_THREADS - 1) /
                           PADDLE_CUDA_NUM_THREADS,
                       PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        in_data, out_data, numel, shifts_ptr, strides_ptr, sizes_ptr, nums);
  }
};

template <typename DeviceContext, typename T>
class RollGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* out = context.Output<LoDTensor>(framework::GradVarName("X"));
    std::vector<int64_t> shifts = context.Attr<std::vector<int64_t>>("shifts");
    std::vector<int64_t> dims = context.Attr<std::vector<int64_t>>("axis");

    auto* in_data = in->data<T>();
    auto* out_data = out->mutable_data<T>(context.GetPlace());
    int64_t numel = in->numel();
    auto stream =
        context.template device_context<platform::CUDADeviceContext>().stream();
    size_t nums = shifts.size();
    auto input_dim = in->dims();
    auto stride_dim = framework::stride(input_dim);

    int64_t dim, size;
    size_t gpu_memory_size_ = sizeof(int64_t) * nums;
    std::vector<int64_t> strides, sizes;
    strides.resize(nums);
    sizes.resize(nums);
    paddle::memory::AllocationPtr shifts_gpu =
        memory::Alloc(context.GetPlace(), gpu_memory_size_);
    paddle::memory::AllocationPtr strides_gpu =
        memory::Alloc(context.GetPlace(), gpu_memory_size_);
    paddle::memory::AllocationPtr sizes_gpu =
        memory::Alloc(context.GetPlace(), gpu_memory_size_);

    for (size_t i = 0; i < nums; i++) {
      dim = dims[i] >= 0 ? dims[i] : dims[i] + input_dim.size();
      size = input_dim[dim];
      shifts[i] = ((0 - shifts[i]) % size + size) % size;
      strides[i] = stride_dim[dim];
      sizes[i] = size;
    }

    paddle::memory::Copy(
        BOOST_GET_CONST(platform::CUDAPlace, shifts_gpu->place()),
        shifts_gpu->ptr(), platform::CPUPlace(), shifts.data(),
        gpu_memory_size_, stream);
    paddle::memory::Copy(
        BOOST_GET_CONST(platform::CUDAPlace, strides_gpu->place()),
        strides_gpu->ptr(), platform::CPUPlace(), strides.data(),
        gpu_memory_size_, stream);
    paddle::memory::Copy(
        BOOST_GET_CONST(platform::CUDAPlace, sizes_gpu->place()),
        sizes_gpu->ptr(), platform::CPUPlace(), sizes.data(), gpu_memory_size_,
        stream);
    int64_t* shifts_ptr = reinterpret_cast<int64_t*>(shifts_gpu->ptr());
    int64_t* strides_ptr = reinterpret_cast<int64_t*>(strides_gpu->ptr());
    int64_t* sizes_ptr = reinterpret_cast<int64_t*>(sizes_gpu->ptr());

    roll_cuda_kernel<<<(numel + PADDLE_CUDA_NUM_THREADS - 1) /
                           PADDLE_CUDA_NUM_THREADS,
                       PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        in_data, out_data, numel, shifts_ptr, strides_ptr, sizes_ptr, nums);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    roll, ops::RollCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::RollCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::RollCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::RollCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    roll_grad,
    ops::RollGradCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::RollGradCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::RollGradCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::RollGradCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
