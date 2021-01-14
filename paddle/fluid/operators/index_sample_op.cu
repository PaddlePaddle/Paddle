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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/index_sample_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T, typename IndexT = int>
__global__ void index_kernel(const IndexT* p_index, const T* p_input,
                             T* p_output, size_t stride_index,
                             size_t stride_input, size_t height) {
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int tid = iy * stride_index + ix;
  int tid_x = iy * stride_input + ix;
  int tid_y = iy * stride_index + ix;

  if (ix < stride_index & iy < height) {
    IndexT idx = p_index[tid];
    p_output[tid_y] = p_input[tid_x - ix + idx];
  }
}

template <typename T, typename IndexT = int>
__global__ void index_kernel_grad(const IndexT_* p_index, T* p_input,
                                  const T* p_output, size_t stride_index,
                                  size_t stride_input, size_t height) {
  extern __shared__ T s_buf[];
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int tid = iy * stride_index + ix;
  int tid_y = iy * stride_input + ix;
  s_buf[tid_y] = p_input[tid_y];
  s_buf[tid_y] = 0;

  if (ix < stride_index & iy < height) {
    for (int i = 0; i < stride_index; ++i) {
      if (ix == i) {
        IndexT idx = p_index[tid];
        s_buf[tid_y - ix + idx] += p_output[tid];
      }
    }
    p_input[tid_y] = s_buf[tid_y];
  }
}

template <typename DeviceContext, typename T>
class IndexSampleCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<LoDTensor>("X");
    auto* index = ctx.Input<LoDTensor>("Index");
    auto* output = ctx.Output<LoDTensor>("Out");

    const auto& index_type = index->type();
    bool index_type_match = index_type == framework::proto::VarType::INT64 ||
                            index_type == framework::proto::VarType::INT32;
    PADDLE_ENFORCE_EQ(index_type_match, true,
                      platform::errors::InvalidArgument(
                          "Input(Index) holds the wrong type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(index_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::InvalidArgument("It must use CUDAPlace."));

    auto ComputeBlockSize = [](int col) {
      if (col > 512)
        return 1024;
      else if (col > 256)
        return 512;
      else if (col > 128)
        return 256;
      else if (col > 64)
        return 128;
      else if (col > 32)
        return 64;
      else if (col > 16)
        return 32;
      else if (col > 8)
        return 16;
      else
        return 8;
    };
    const auto* in_data = input->data<T>();
    auto* out_data = output->mutable_data<T>(ctx.GetPlace());
    auto stream =
        ctx.template device_context<platform::CUDADeviceContext>().stream();

    auto input_dim = input->dims();
    auto index_dim = index->dims();
    size_t batch_size = input_dim[0];
    size_t input_length = input_dim[1];
    size_t index_length = index_dim[1];

    auto block_width = ComputeBlockSize(index_length);
    int block_height =
        ComputeBlockSize(index_length * batch_size) / block_width;

    dim3 block_dim(block_width, block_height);
    dim3 grid_dim((index_length + block_dim.x - 1) / block_dim.x,
                  (batch_size + block_dim.y - 1) / block_dim.y);

    if (index_type == framework::proto::VarType::INT64) {
      const int64_t* index_data = index->data<int64_t>();
      index_kernel<T, int64_t><<<grid_dim, block_dim, 0, stream>>>(
          index_data, in_data, out_data, index_length, input_length,
          batch_size);
    } else if (index_type == framework::proto::VarType::INT32) {
      const int* index_data = index->data<int>();
      index_kernel<T, int><<<grid_dim, block_dim, 0, stream>>>(
          index_data, in_data, out_data, index_length, input_length,
          batch_size);
    }
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  }
};

template <typename DeviceContext, typename T>
class IndexSampleGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* output_grad = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* input_grad = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    auto* index = ctx.Input<LoDTensor>("Index");

    const auto* output_grad_data = output_grad->data<T>();
    auto* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());

    const auto& index_type = index->type();
    bool index_type_match = index_type == framework::proto::VarType::INT64 ||
                            index_type == framework::proto::VarType::INT32;
    PADDLE_ENFORCE_EQ(index_type_match, true,
                      platform::errors::InvalidArgument(
                          "Input(Index) holds the wrong type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(index_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::InvalidArgument("It must use CUDAPlace."));

    auto ComputeBlockSize = [](int col) {
      if (col > 512)
        return 1024;
      else if (col > 256)
        return 512;
      else if (col > 128)
        return 256;
      else if (col > 64)
        return 128;
      else if (col > 32)
        return 64;
      else if (col > 16)
        return 32;
      else if (col > 8)
        return 16;
      else
        return 8;
    };
    auto stream =
        ctx.template device_context<platform::CUDADeviceContext>().stream();

    auto input_num = input_grad->numel();
    auto input_dim = input_grad->dims();
    auto index_dim = index->dims();
    size_t batch_size = index_dim[0];
    size_t input_length = input_dim[1];
    size_t index_length = index_dim[1];

    auto block_width = ComputeBlockSize(index_length);
    auto block_height =
        ComputeBlockSize(index_length * batch_size) / block_width;

    dim3 block_dim(block_width, block_height);
    dim3 grid_dim((index_length + block_dim.x - 1) / block_dim.x,
                  (batch_size + block_dim.y - 1) / block_dim.y);

    if (index_type == framework::proto::VarType::INT64) {
      const int64_t* index_data = index->data<int64_t>();
      index_kernel_grad<
          T, int64_t><<<grid_dim, block_dim, input_num * sizeof(T), stream>>>(
          index_data, output_grad_data, input_grad_data, index_length,
          input_length, batch_size);
    } else if (index_type == framework::proto::VarType::INT32) {
      const int* index_data = index->data<int>();
      index_kernel_grad<
          T, int><<<grid_dim, block_dim, input_num * sizeof(T), stream>>>(
          index_data, output_grad_data, input_grad_data, index_length,
          input_length, batch_size);
    }
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    index_sample,
    ops::IndexSampleCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::IndexSampleCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::IndexSampleCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::IndexSampleCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    index_sample_grad,
    ops::IndexSampleGradCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::IndexSampleGradCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::IndexSampleGradCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::IndexSampleGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                   int64_t>);
