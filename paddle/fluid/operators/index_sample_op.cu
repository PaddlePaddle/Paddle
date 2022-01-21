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
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

#define PREDEFINED_BLOCK_SIZE_X 512
#define PREDEFINED_BLOCK_SIZE 1024
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace paddle {
namespace operators {

namespace {
void LimitGridDim(const framework::ExecutionContext& ctx, dim3* grid_dim) {
  dim3 max_grid_dim = ctx.template device_context<platform::CUDADeviceContext>()
                          .GetCUDAMaxGridDimSize();
  grid_dim->x = grid_dim->x < max_grid_dim.x ? grid_dim->x : max_grid_dim.x;
  grid_dim->y = grid_dim->y < max_grid_dim.y ? grid_dim->y : max_grid_dim.y;
}
}

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T, typename IndexT = int>
__global__ void IndexSampleForward(const IndexT* index, const T* in_data,
                                   T* out_data, size_t index_length,
                                   size_t input_length, size_t batch_size) {
  unsigned int index_i = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int index_j = blockDim.y * blockIdx.y + threadIdx.y;
  for (; index_j < batch_size; index_j += blockDim.y * gridDim.y) {
    for (; index_i < index_length; index_i += blockDim.x * gridDim.x) {
      unsigned int index_idx = index_j * index_length + index_i;
      unsigned int in_idx = index_j * input_length + index_i;
      IndexT sample_idx = index[index_idx];
      out_data[index_idx] = in_data[in_idx - index_i + sample_idx];
    }
  }
}

template <typename T, typename IndexT = int>
__global__ void IndexSampleGrad(const IndexT* index, T* in_grad,
                                const T* out_grad, size_t index_length,
                                size_t input_length, size_t batch_size,
                                bool same_data_in_row = true) {
  unsigned int index_i = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int index_j = blockDim.y * blockIdx.y + threadIdx.y;

  for (; index_j < batch_size; index_j += blockDim.y * gridDim.y) {
    for (; index_i < index_length; index_i += blockDim.x * gridDim.x) {
      unsigned int index_idx = index_j * index_length + index_i;
      unsigned int in_idx = index_j * input_length + index_i;
      IndexT sample_idx = index[index_idx];
      if (same_data_in_row) {
        platform::CudaAtomicAdd(&(in_grad[in_idx - index_i + sample_idx]),
                                out_grad[sample_idx]);
      } else {
        in_grad[in_idx - index_i + sample_idx] = out_grad[index_idx];
      }
    }
  }
}

template <typename T>
class IndexSampleKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
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
    block_width = MIN(block_width, PREDEFINED_BLOCK_SIZE_X);
    int block_height =
        platform::RoundToPowerOfTwo(index_length * batch_size) / block_width;
    block_height = MIN(block_height, PREDEFINED_BLOCK_SIZE / block_width);
    dim3 block_dim(block_width, block_height);
    dim3 grid_dim((index_length + block_dim.x - 1) / block_dim.x,
                  (batch_size + block_dim.y - 1) / block_dim.y);
    LimitGridDim(ctx, &grid_dim);

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
  }
};

template <typename T>
class IndexSampleGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
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

    auto stream =
        ctx.template device_context<platform::CUDADeviceContext>().stream();
    auto input_num = input_grad->numel();
    auto input_dim = input_grad->dims();
    auto index_dim = index->dims();
    size_t batch_size = index_dim[0];
    size_t input_length = input_dim[1];
    size_t index_length = index_dim[1];
    bool same_data_in_index_row = index_length == 1 ? false : true;

    auto block_width = platform::RoundToPowerOfTwo(index_length);
    block_width = MIN(block_width, PREDEFINED_BLOCK_SIZE_X);
    auto block_height =
        platform::RoundToPowerOfTwo(index_length * batch_size) / block_width;
    block_height = MIN(block_height, PREDEFINED_BLOCK_SIZE / block_width);
    dim3 block_dim(block_width, block_height);
    dim3 grid_dim((index_length + block_dim.x - 1) / block_dim.x,
                  (batch_size + block_dim.y - 1) / block_dim.y);
    LimitGridDim(ctx, &grid_dim);

    math::SetConstant<platform::CUDADeviceContext, T> set_zero;
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    set_zero(dev_ctx, input_grad, static_cast<T>(0));

    if (index_type == framework::proto::VarType::INT64) {
      const int64_t* index_data = index->data<int64_t>();
      IndexSampleGrad<T, int64_t><<<grid_dim, block_dim, 0, stream>>>(
          index_data, input_grad_data, output_grad_data, index_length,
          input_length, batch_size, same_data_in_index_row);
    } else if (index_type == framework::proto::VarType::INT32) {
      const int* index_data = index->data<int>();
      IndexSampleGrad<T, int><<<grid_dim, block_dim, 0, stream>>>(
          index_data, input_grad_data, output_grad_data, index_length,
          input_length, batch_size, same_data_in_index_row);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    index_sample,
    ops::IndexSampleKernel<paddle::platform::CUDADeviceContext, float>,
    ops::IndexSampleKernel<paddle::platform::CUDADeviceContext, double>,
    ops::IndexSampleKernel<paddle::platform::CUDADeviceContext, int>,
    ops::IndexSampleKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    index_sample_grad,
    ops::IndexSampleGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::IndexSampleGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::IndexSampleGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::IndexSampleGradKernel<paddle::platform::CUDADeviceContext, int64_t>);
