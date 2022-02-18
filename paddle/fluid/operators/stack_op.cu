// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <limits>
#include <vector>
#include "paddle/fluid/operators/stack_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"

namespace plat = paddle::platform;
namespace ops = paddle::operators;

namespace paddle {
namespace operators {

template <typename T, typename IntType>
__global__ void StackCUDAKernel(T** input_ptrs, int split_size, int rows,
                                int cols, T* __restrict__ output) {
  IntType grid_x = blockIdx.x * blockDim.x + threadIdx.x;

  for (; grid_x < cols; grid_x += blockDim.x * gridDim.x) {
    IntType grid_y = blockIdx.y * blockDim.y + threadIdx.y;

    IntType split = grid_x / split_size;
    const T* input_ptr = input_ptrs[split];
    IntType col_offset = grid_x % split_size;
#pragma unroll
    for (; grid_y < rows; grid_y += blockDim.y * gridDim.y) {
      output[grid_y * cols + grid_x] =
          input_ptr[grid_y * split_size + col_offset];
    }
  }
}

template <typename T>
class StackGPUKernel : public framework::OpKernel<T> {
  using Tensor = framework::LoDTensor;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.MultiInput<Tensor>("X");
    auto* y = ctx.Output<Tensor>("Y");

    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis += (x[0]->dims().size() + 1);

    int n = static_cast<int>(x.size());
    auto* y_data = y->mutable_data<T>(ctx.GetPlace());
    std::vector<const T*> x_datas(n);
    for (int i = 0; i < n; i++) {
      x_datas[i] = x[i]->data<T>();
    }

    auto& dev_ctx = ctx.template device_context<plat::CUDADeviceContext>();
    auto tmp_x_data = memory::Alloc(dev_ctx, x_datas.size() * sizeof(T*));
    memory::Copy(dev_ctx.GetPlace(), tmp_x_data->ptr(), platform::CPUPlace(),
                 reinterpret_cast<void*>(x_datas.data()),
                 x_datas.size() * sizeof(T*), dev_ctx.stream());

    // Split x dim from axis to matrix
    int x_row = 1, x_col = 1;
    for (int i = 0; i < axis; ++i) {
      x_row *= x[0]->dims()[i];
    }
    x_col = x[0]->numel() / x_row;
    int out_col = x_col * n;

    auto config = GetGpuLaunchConfig2D(dev_ctx, out_col, x_row);

    if (y->numel() < std::numeric_limits<int32_t>::max()) {
      StackCUDAKernel<T,
                      int32_t><<<config.block_per_grid, config.thread_per_block,
                                 0, dev_ctx.stream()>>>(
          reinterpret_cast<T**>(tmp_x_data->ptr()), x_col, x_row, out_col,
          y_data);
    } else {
      StackCUDAKernel<T,
                      int64_t><<<config.block_per_grid, config.thread_per_block,
                                 0, dev_ctx.stream()>>>(
          reinterpret_cast<T**>(tmp_x_data->ptr()), x_col, x_row, out_col,
          y_data);
    }
  }
};

template <typename T, typename IntType>
__global__ void UnStackHelperCUDAKernel(const T* __restrict__ input,
                                        int pre_dim_size, int split_dim_size,
                                        int suf_dim_size, int num_split,
                                        T** output_ptrs) {
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  // In this case they are equal
  assert(split_dim_size % num_split == 0);

  IntType size = pre_dim_size * split_dim_size * suf_dim_size;
  IntType each_dim_size = split_dim_size / num_split;

  for (IntType offset = blockIdx.x * blockDim.x + threadIdx.x; offset < size;
       offset += blockDim.x * gridDim.x) {
    IntType i = offset / (split_dim_size * suf_dim_size);
    IntType j = (offset % (split_dim_size * suf_dim_size)) / suf_dim_size;
    IntType k = offset % suf_dim_size;

    T* output = output_ptrs[j / each_dim_size];
    if (output == nullptr) {
      return;
    }
    IntType output_ind = i * each_dim_size * suf_dim_size +
                         (j % each_dim_size) * suf_dim_size + k;
    *(output + output_ind) = input[offset];
  }
}

template <typename T>
class StackGradGPUKernel : public framework::OpKernel<T> {
  using Tensor = framework::LoDTensor;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto dx = ctx.MultiOutput<Tensor>(framework::GradVarName("X"));
    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis += dy->dims().size();

    int n = dy->dims()[axis];
    PADDLE_ENFORCE_EQ(n, dx.size(),
                      platform::errors::InvalidArgument(
                          "Output dx size should be equal to n, but"
                          " received n is:%d dx size is:%d.",
                          n, dx.size()));

    // dx is output, so save each data address, then copy each dy into dx_data
    std::vector<T*> outputs(n);
    auto out_var_names = ctx.OutputNames(framework::GradVarName("X"));
    for (size_t j = 0; j < dx.size(); ++j) {
      if (dx[j] == nullptr) {
        outputs[j] = nullptr;
      }
      if (out_var_names[j] != framework::kEmptyVarName &&
          dx[j]->numel() != 0UL) {
        T* ptr = dx[j]->mutable_data<T>(ctx.GetPlace());
        outputs[j] = ptr;
      } else {
        outputs[j] = nullptr;
      }
    }
    auto dy_data = dy->data<T>();
    // each dx should have same shape
    int dy_pre = 1, dy_suf = 1;
    auto dy_dims = dy->dims();
    int split_dim = n;
    for (int i = 0; i < axis; ++i) {
      dy_pre *= dy_dims[i];
    }
    dy_suf = dy->numel() / (split_dim * dy_pre);

    auto& dev_ctx = ctx.template device_context<plat::CUDADeviceContext>();
    auto tmp_out_data = memory::Alloc(dev_ctx, outputs.size() * sizeof(T*));
    memory::Copy(dev_ctx.GetPlace(), tmp_out_data->ptr(), platform::CPUPlace(),
                 reinterpret_cast<void*>(outputs.data()),
                 outputs.size() * sizeof(T*), dev_ctx.stream());

    auto config = GetGpuLaunchConfig1D(dev_ctx, dy_pre * split_dim * dy_suf);

    if (dy->numel() < std::numeric_limits<int32_t>::max()) {
      UnStackHelperCUDAKernel<
          T, int32_t><<<config.block_per_grid.x, config.thread_per_block.x, 0,
                        dev_ctx.stream()>>>(
          dy_data, dy_pre, split_dim, dy_suf, split_dim,
          reinterpret_cast<T**>(tmp_out_data->ptr()));
    } else {
      UnStackHelperCUDAKernel<
          T, int64_t><<<config.block_per_grid.x, config.thread_per_block.x, 0,
                        dev_ctx.stream()>>>(
          dy_data, dy_pre, split_dim, dy_suf, split_dim,
          reinterpret_cast<T**>(tmp_out_data->ptr()));
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(stack, ops::StackGPUKernel<float>,
                        ops::StackGPUKernel<double>, ops::StackGPUKernel<int>,
                        ops::StackGPUKernel<int64_t>,
                        ops::StackGPUKernel<plat::float16>,
                        ops::StackGPUKernel<plat::bfloat16>);

REGISTER_OP_CUDA_KERNEL(stack_grad, ops::StackGradGPUKernel<float>,
                        ops::StackGradGPUKernel<double>,
                        ops::StackGradGPUKernel<int>,
                        ops::StackGradGPUKernel<int64_t>,
                        ops::StackGradGPUKernel<plat::float16>,
                        ops::StackGradGPUKernel<plat::bfloat16>);
