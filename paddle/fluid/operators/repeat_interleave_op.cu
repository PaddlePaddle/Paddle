// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/repeat_interleave_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

// function borrowed from repeat_interleave_op
template <typename T, typename IndexT>
__global__ void index_select_cuda_kernel(const T* input, T* output,
                                         const IndexT* index, int64_t N,
                                         int64_t stride, int64_t size,
                                         int64_t delta) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  int64_t pre_idx = idx / (stride * size);
  int64_t dim_idx = idx % (stride * size) / stride;
  IndexT src_dim_idx = index[dim_idx];
  int64_t input_idx = idx + (delta * pre_idx + src_dim_idx - dim_idx) * stride;
  output[idx] = input[input_idx];
}

template <typename T, typename IndexT>
__global__ void index_select_grad_cuda_kernel(const T* output_grad,
                                              T* input_grad,
                                              const IndexT* index, int64_t nums,
                                              int64_t N, int64_t stride,
                                              int64_t size, int64_t delta) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  int64_t pre_idx = idx / (stride * size);
  int64_t dim_idx = idx % (stride * size) / stride;
  IndexT src_dim_idx = index[dim_idx];
  int64_t input_idx = idx + (delta * pre_idx + src_dim_idx - dim_idx) * stride;
  paddle::platform::CudaAtomicAdd(&input_grad[input_idx], output_grad[idx]);
}

template <typename T>
__global__ void index_select_grad_init(T* input_grad, int64_t N) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }
  input_grad[idx] = 0.0;
}
template <typename DeviceContext, typename T>
class RepeatInterleaveCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    // auto* index = context.Input<LoDTensor>("RepeatsTensor");
    auto* out = context.Output<LoDTensor>("Out");
    int dim = context.Attr<int>("dim");
    auto input_dim = in->dims();
    dim = dim >= 0 ? dim : dim + input_dim.size();
    auto stride_dim = framework::stride(input_dim);
    int64_t stride = stride_dim[dim];

    auto stream =
        context.template device_context<platform::CUDADeviceContext>().stream();

    int repeats = context.Attr<int>("Repeats");
    framework::LoDTensor index;
    auto* in_data = in->data<T>();
    if (context.HasInput("RepeatsTensor")) {
      auto repeats_tensor =
          context.Input<framework::LoDTensor>("RepeatsTensor");

      PADDLE_ENFORCE_EQ(repeats_tensor->dims()[0] == in->dims()[dim], true,
                        platform::errors::InvalidArgument(
                            "The length of Input(RepeatsTensor) must be the "
                            "same as length of Input(X) in axis. "
                            "But received: [%s], required: [%d].",
                            repeats_tensor->dims()[0], in->dims()[dim]));

      const auto& index_type = repeats_tensor->type();
      bool index_type_match = index_type == framework::proto::VarType::INT64 ||
                              index_type == framework::proto::VarType::INT32;
      PADDLE_ENFORCE_EQ(
          index_type_match, true,
          platform::errors::InvalidArgument(
              "Input(RepeatsTensor) holds the wrong type, it holds %s, but "
              "desires to be %s or %s",
              paddle::framework::DataTypeToString(index_type),
              paddle::framework::DataTypeToString(
                  framework::proto::VarType::INT32),
              paddle::framework::DataTypeToString(
                  framework::proto::VarType::INT64)));

      if (index_type == framework::proto::VarType::INT64) {
        RepeatsTensor2IndexTensor<DeviceContext, int64_t>(*repeats_tensor,
                                                          &index);

        const int64_t* index_data = index.data<int64_t>();
        auto output_dim = framework::vectorize(in->dims());
        output_dim[dim] = index.dims()[0];
        out->Resize(framework::make_ddim(output_dim));
        auto* out_data = out->mutable_data<T>(context.GetPlace());
        int64_t numel = out->numel();
        int64_t size = output_dim[dim];
        int64_t delta = input_dim[dim] - size;

        index_select_cuda_kernel<T, int64_t><<<
            (numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
            PADDLE_CUDA_NUM_THREADS, 0, stream>>>(in_data, out_data, index_data,
                                                  numel, stride, size, delta);
      } else {
        RepeatsTensor2IndexTensor<DeviceContext, int>(*repeats_tensor, &index);

        const int* index_data = index.data<int>();
        auto output_dim = framework::vectorize(in->dims());
        output_dim[dim] = index.dims()[0];
        out->Resize(framework::make_ddim(output_dim));
        auto* out_data = out->mutable_data<T>(context.GetPlace());
        int64_t numel = out->numel();
        int64_t size = output_dim[dim];
        int64_t delta = input_dim[dim] - size;

        index_select_cuda_kernel<T, int><<<
            (numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
            PADDLE_CUDA_NUM_THREADS, 0, stream>>>(in_data, out_data, index_data,
                                                  numel, stride, size, delta);
      }
    } else if (repeats > 0) {
      int64_t index_size = in->dims()[dim] * repeats;
      std::vector<int> index_vec(index_size);
      for (int i = 0; i < in->dims()[dim]; i++) {
        std::fill_n(index_vec.begin() + i * repeats, repeats, i);
      }
      index.Resize(framework::make_ddim({index_size}));
      auto ctx = paddle::platform::DeviceContextPool::Instance().Get(
          context.GetPlace());
      paddle::framework::TensorFromVector<int>(index_vec, *ctx, &index);

      auto output_dim = framework::vectorize(in->dims());
      output_dim[dim] = index_size;
      out->Resize(framework::make_ddim(output_dim));
      auto* out_data = out->mutable_data<T>(context.GetPlace());

      int64_t numel = out->numel();
      int64_t size = output_dim[dim];
      int64_t delta = input_dim[dim] - size;

      const int* index_data = index.data<int>();
      index_select_cuda_kernel<T, int><<<(numel + PADDLE_CUDA_NUM_THREADS - 1) /
                                             PADDLE_CUDA_NUM_THREADS,
                                         PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
          in_data, out_data, index_data, numel, stride, size, delta);
      platform::GpuStreamSync(stream);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "repeats must given with RepeatsTensor (tensor) or repeats (int)"));
    }
  }
};

template <typename DeviceContext, typename T>
class RepeatInterleaveGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* output_grad = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* in_grad = context.Output<LoDTensor>(framework::GradVarName("X"));

    auto* output_grad_data = output_grad->data<T>();
    auto* in_grad_data = in_grad->mutable_data<T>(context.GetPlace());

    int dim = context.Attr<int>("dim");
    auto input_dim = in_grad->dims();
    auto output_dim = output_grad->dims();
    dim = dim >= 0 ? dim : dim + input_dim.size();
    auto stride_dim = framework::stride(input_dim);
    int64_t stride = stride_dim[dim];
    int64_t size = output_dim[dim];
    int64_t delta = input_dim[dim] - size;

    int64_t numel = in_grad->numel();
    int64_t out_nums = output_grad->numel();

    auto stream =
        context.template device_context<platform::CUDADeviceContext>().stream();

    index_select_grad_init<
        T><<<(numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
             PADDLE_CUDA_NUM_THREADS, 0, stream>>>(in_grad_data, numel);

    int repeats = context.Attr<int>("Repeats");
    framework::LoDTensor index;
    if (context.HasInput("RepeatsTensor")) {
      auto repeats_tensor =
          context.Input<framework::LoDTensor>("RepeatsTensor");

      const auto& index_type = repeats_tensor->type();
      bool index_type_match = index_type == framework::proto::VarType::INT64 ||
                              index_type == framework::proto::VarType::INT32;
      PADDLE_ENFORCE_EQ(
          index_type_match, true,
          platform::errors::InvalidArgument(
              "Input(Index) holds the wrong type, it holds %s, but "
              "desires to be %s or %s",
              paddle::framework::DataTypeToString(index_type),
              paddle::framework::DataTypeToString(
                  framework::proto::VarType::INT32),
              paddle::framework::DataTypeToString(
                  framework::proto::VarType::INT64)));

      if (index_type == framework::proto::VarType::INT64) {
        RepeatsTensor2IndexTensor<DeviceContext, int64_t>(*repeats_tensor,
                                                          &index);
        int64_t index_nums = index.numel();

        const int64_t* index_data = index.data<int64_t>();
        index_select_grad_cuda_kernel<T, int64_t><<<
            (out_nums + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
            PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
            output_grad_data, in_grad_data, index_data, index_nums, out_nums,
            stride, size, delta);
        platform::GpuStreamSync(stream);
      } else {
        RepeatsTensor2IndexTensor<DeviceContext, int>(*repeats_tensor, &index);
        int64_t index_nums = index.numel();

        const int* index_data = index.data<int>();
        index_select_grad_cuda_kernel<T, int><<<
            (out_nums + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
            PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
            output_grad_data, in_grad_data, index_data, index_nums, out_nums,
            stride, size, delta);
        platform::GpuStreamSync(stream);
      }
    } else if (repeats > 0) {
      int64_t index_size = in_grad->dims()[dim] * repeats;
      std::vector<int> index_vec(index_size);
      for (int i = 0; i < in_grad->dims()[dim]; i++) {
        std::fill_n(index_vec.begin() + i * repeats, repeats, i);
      }
      index.Resize(framework::make_ddim({index_size}));
      auto ctx = paddle::platform::DeviceContextPool::Instance().Get(
          context.GetPlace());
      paddle::framework::TensorFromVector<int>(index_vec, *ctx, &index);

      const int* index_data = index.data<int>();
      int64_t index_nums = index.numel();
      index_select_grad_cuda_kernel<T, int><<<
          (out_nums + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
          PADDLE_CUDA_NUM_THREADS, 0, stream>>>(output_grad_data, in_grad_data,
                                                index_data, index_nums,
                                                out_nums, stride, size, delta);
      platform::GpuStreamSync(stream);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "repeats must given with RepeatsTensor (tensor) or repeats (int)"));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    repeat_interleave,
    ops::RepeatInterleaveCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::RepeatInterleaveCUDAKernel<paddle::platform::CUDADeviceContext,
                                    double>,
    ops::RepeatInterleaveCUDAKernel<paddle::platform::CUDADeviceContext,
                                    paddle::platform::float16>,
    ops::RepeatInterleaveCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::RepeatInterleaveCUDAKernel<paddle::platform::CUDADeviceContext,
                                    int64_t>);
REGISTER_OP_CUDA_KERNEL(
    repeat_interleave_grad,
    ops::RepeatInterleaveGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                        float>,
    ops::RepeatInterleaveGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                        double>,
    ops::RepeatInterleaveGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                        paddle::platform::float16>,
    ops::RepeatInterleaveGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                        int>,
    ops::RepeatInterleaveGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                        int64_t>);
