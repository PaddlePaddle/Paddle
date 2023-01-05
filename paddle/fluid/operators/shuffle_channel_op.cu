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

#include "paddle/fluid/operators/shuffle_channel_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <typename T>
__global__ void ShuffleChannel(const int nthreads,
                               const int feature_map_size,
                               T* output,
                               const T* input,
                               int group_row,
                               int group_column,
                               int len) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (size_t ii = index; ii < nthreads; ii += offset) {
    const int n = index / group_row / group_column / len;
    const int i = (index / group_column / len) % group_row;
    const int j = index / len % group_column;
    const int k = index - (n * feature_map_size + (i * group_column + j) * len);
    T* p_o = output + n * feature_map_size + (j * group_row + i) * len;
    p_o[k] = input[index];
  }
}
template <typename DeviceContext, typename T>
class ShuffleChannelOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* output = ctx.Output<phi::DenseTensor>("Out");
    int group = ctx.Attr<int>("group");

    auto input_dims = input->dims();
    auto num = input_dims[0];
    auto channel = input_dims[1];
    auto height = input_dims[2];
    auto weight = input_dims[3];

    auto feature_map_size = channel * height * weight;
    auto sp_sz = height * weight;
    int group_row = group;
    int group_column = channel / group_row;
    // count is the product of NCHW same as numel()
    int count = num * group_column * group_row * sp_sz;

    int blocks = NumBlocks(output->numel());
    int threads = kNumCUDAThreads;

    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    ShuffleChannel<T>
        <<<blocks, threads, 0, ctx.cuda_device_context().stream()>>>(
            count,
            feature_map_size,
            output_data,
            input_data,
            group_row,
            group_column,
            sp_sz);
  }
};

template <typename DeviceContext, typename T>
class ShuffleChannelGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* output_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* input_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));

    int group = ctx.Attr<int>("group");

    const auto& input_dims = input_grad->dims();
    auto num = input_dims[0];
    auto channel = input_dims[1];
    auto height = input_dims[2];
    auto weight = input_dims[3];
    auto feature_map_size = channel * height * weight;
    auto sp_sz = height * weight;

    int group_row = group;
    int group_column = channel / group_row;

    T* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
    const T* output_grad_data = output_grad->data<T>();

    int blocks = NumBlocks(output_grad->numel());
    int threads = kNumCUDAThreads;
    int count = num * group_column * group_row * sp_sz;

    ShuffleChannel<T>
        <<<blocks, threads, 0, ctx.cuda_device_context().stream()>>>(
            count,
            feature_map_size,
            input_grad_data,
            output_grad_data,
            group_row,
            group_column,
            sp_sz);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    shuffle_channel,
    ops::ShuffleChannelOpCUDAKernel<phi::GPUContext, float>,
    ops::ShuffleChannelOpCUDAKernel<phi::GPUContext, double>);
REGISTER_OP_CUDA_KERNEL(
    shuffle_channel_grad,
    ops::ShuffleChannelGradOpCUDAKernel<phi::GPUContext, float>,
    ops::ShuffleChannelGradOpCUDAKernel<phi::GPUContext, double>);
