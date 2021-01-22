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
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T, typename IndexT = int>
__global__ void IndexSampleForward(const IndexT* index, const T* in_data,
                                   T* out_data, size_t index_length,
                                   size_t input_length, size_t batch_size) {
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int tid = iy * index_length + ix;
  int tid_x = iy * input_length + ix;

  if (ix < index_length & iy < batch_size) {
    IndexT idx = index[tid];
    out_data[tid] = in_data[tid_x - ix + idx];
  }
}

template <typename T, typename IndexT = int>
__global__ void IndexSampleGradDefault(const IndexT* index, T* in_grad,
                                       const T* out_grad, size_t index_length,
                                       size_t input_length, size_t batch_size) {
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int tid = iy * index_length + ix;
  int tid_y = iy * input_length + ix;

  if (ix < index_length & iy < batch_size) {
    IndexT idx = index[tid];
    platform::CudaAtomicAdd(&(in_grad[tid_y - ix + idx]), out_grad[tid]);
  }
}

template <typename T, typename IndexT = int>
__global__ void IndexSampleGradSpecial(const IndexT* index, T* in_grad,
                                       const T* out_grad, size_t index_length,
                                       size_t input_length, size_t batch_size) {
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  int tid = iy * index_length + ix;
  int tid_y = iy * input_length + ix;

  if (ix < index_length & iy < batch_size) {
    IndexT idx = index[tid];
    in_grad[tid_y - ix + idx] = out_grad[tid];
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

    const auto* in_data = input->data<T>();
    auto* out_data = output->mutable_data<T>(ctx.GetPlace());
    auto stream =
        ctx.template device_context<platform::CUDADeviceContext>().stream();

    auto input_dim = input->dims();
    auto index_dim = index->dims();
    size_t batch_size = input_dim[0];
    size_t input_length = input_dim[1];
    size_t index_length = index_dim[1];

    auto block_width = platform::RoundToPowerOfTwo(index_length);
    int block_height =
        platform::RoundToPowerOfTwo(index_length * batch_size) / block_width;

    dim3 block_dim(block_width, block_height);
    dim3 grid_dim((index_length + block_dim.x - 1) / block_dim.x,
                  (batch_size + block_dim.y - 1) / block_dim.y);

    if (index_type == framework::proto::VarType::INT64) {
      const int64_t* index_data = index->data<int64_t>();
      IndexSampleForward<T, int64_t><<<grid_dim, block_dim, 0, stream>>>(
          index_data, in_data, out_data, index_length, input_length,
          batch_size);
    } else if (index_type == framework::proto::VarType::INT32) {
      const int* index_data = index->data<int>();
      IndexSampleForward<T, int><<<grid_dim, block_dim, 0, stream>>>(
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

    auto stream =
        ctx.template device_context<platform::CUDADeviceContext>().stream();

    auto input_num = input_grad->numel();
    auto input_dim = input_grad->dims();
    auto index_dim = index->dims();
    size_t batch_size = index_dim[0];
    size_t input_length = input_dim[1];
    size_t index_length = index_dim[1];

    auto block_width = platform::RoundToPowerOfTwo(index_length);
    auto block_height =
        platform::RoundToPowerOfTwo(index_length * batch_size) / block_width;

    dim3 block_dim(block_width, block_height);
    dim3 grid_dim((index_length + block_dim.x - 1) / block_dim.x,
                  (batch_size + block_dim.y - 1) / block_dim.y);

    platform::GpuMemsetAsync(input_grad_data, 0,
                             sizeof(T) * input_length * batch_size, stream);

    if (index_type == framework::proto::VarType::INT64) {
      const int64_t* index_data = index->data<int64_t>();
      if (index_length == 1) {
        IndexSampleGradSpecial<T, int64_t><<<grid_dim, block_dim, 0, stream>>>(
            index_data, input_grad_data, output_grad_data, index_length,
            input_length, batch_size);
      } else {
        IndexSampleGradDefault<T, int64_t><<<grid_dim, block_dim, 0, stream>>>(
            index_data, input_grad_data, output_grad_data, index_length,
            input_length, batch_size);
      }
    } else if (index_type == framework::proto::VarType::INT32) {
      const int* index_data = index->data<int>();
      if (index_length == 1) {
        IndexSampleGradSpecial<T, int><<<grid_dim, block_dim, 0, stream>>>(
            index_data, input_grad_data, output_grad_data, index_length,
            input_length, batch_size);
      } else {
        IndexSampleGradDefault<T, int><<<grid_dim, block_dim, 0, stream>>>(
            index_data, input_grad_data, output_grad_data, index_length,
            input_length, batch_size);
      }
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
