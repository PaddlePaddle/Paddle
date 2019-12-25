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
#include <vector>
#include "paddle/fluid/operators/stack_op.h"

namespace plat = paddle::platform;
namespace ops = paddle::operators;

namespace paddle {
namespace operators {

static inline void Get2DGpuLaunchConfig(
    const platform::CUDADeviceContext& context, int num_rows, int num_cols,
    dim3* block_dims, dim3* grid_dims) {
  // Set the thread block and grid according to CurrentDeviceId
  const int kThreadsPerBlock = 512;
  int block_cols = kThreadsPerBlock;
  if (num_cols < kThreadsPerBlock) {  // block_cols is aligned by 32.
    block_cols = ((num_cols + 31) >> 5) << 5;
  }
  int block_rows = kThreadsPerBlock / block_cols;
  *block_dims = dim3(block_cols, block_rows, 1);

  int max_threads = context.GetMaxPhysicalThreadCount();
  int max_blocks = std::max(max_threads / kThreadsPerBlock, 1);

  int grid_cols =
      std::min((num_cols + block_cols - 1) / block_cols, max_blocks);
  int grid_rows =
      std::min(max_blocks / grid_cols, std::max(num_rows / block_rows, 1));
  *grid_dims = dim3(grid_cols, grid_rows, 1);
}

template <typename T, typename IntType>
__global__ void stack_kernel(T** input_ptrs, int split_size, int rows, int cols,
                             T* __restrict__ output) {
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

template <typename DeviceContext, typename T>
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

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto tmp_x_data = memory::Alloc(dev_ctx, x_datas.size() * sizeof(T*));
    memory::Copy(boost::get<platform::CUDAPlace>(dev_ctx.GetPlace()),
                 tmp_x_data->ptr(), platform::CPUPlace(),
                 reinterpret_cast<void*>(x_datas.data()),
                 x_datas.size() * sizeof(T*), dev_ctx.stream());

    // Split x dim from axis to matrix
    int x_row = 1, x_col = 1;
    for (int i = 0; i < axis; ++i) {
      x_row *= x[0]->dims()[i];
    }
    x_col = x[0]->numel() / x_row;
    int out_col = x_col * n;

    dim3 block_dims;
    dim3 grid_dims;
    Get2DGpuLaunchConfig(dev_ctx, x_row, out_col, &block_dims, &grid_dims);
    stack_kernel<T, int><<<grid_dims, block_dims, 0, dev_ctx.stream()>>>(
        reinterpret_cast<T**>(tmp_x_data->ptr()), x_col, x_row, out_col,
        y_data);
  }
};

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

static inline void Get1DGpuLaunchConfig(
    const platform::CUDADeviceContext& context, int element_count,
    int* threads_per_block, int* blocks_per_grid) {
  const int theory_thread_count = element_count;
  int max_threads = context.GetMaxPhysicalThreadCount();
  const int physical_thread_count = std::min(max_threads, theory_thread_count);

  // Need get from device
  const int thread_per_block = 1024;
  *threads_per_block = thread_per_block;
  // suppose SM can hold 8 block
  int sm_block = context.GetSMCount() * 8;
  const int block_count =
      std::min(DivUp(physical_thread_count, thread_per_block), sm_block);
  *blocks_per_grid = block_count;
}

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename T>
__global__ void unstack_kernel(const T* __restrict__ input, int pre_dim_size,
                               int split_dim_size, int suf_dim_size,
                               int num_split, T** output_ptrs) {
  eigen_assert(blockDim.y == 1);
  eigen_assert(blockDim.z == 1);
  // In this case they are equal
  eigen_assert(split_dim_size % num_split == 0);

  int size = pre_dim_size * split_dim_size * suf_dim_size;
  int each_dim_size = split_dim_size / num_split;

  CUDA_1D_KERNEL_LOOP(offset, size) {
    int i = offset / (split_dim_size * suf_dim_size);
    int j = (offset % (split_dim_size * suf_dim_size)) / suf_dim_size;
    int k = offset % suf_dim_size;

    T* output = output_ptrs[j / each_dim_size];
    int output_ind = i * each_dim_size * suf_dim_size +
                     (j % each_dim_size) * suf_dim_size + k;
    *(output + output_ind) = input[offset];
  }
}

template <typename DeviceContext, typename T>
class StackGradGPUKernel : public framework::OpKernel<T> {
  using Tensor = framework::LoDTensor;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto dx = ctx.MultiOutput<Tensor>(framework::GradVarName("X"));
    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis += dy->dims().size();

    int n = dy->dims()[axis];
    PADDLE_ENFORCE_EQ(n, dx.size(), "Output dx size should be equal to n");
    // dx is output, so save each data address, then copy each dy into dx_data
    std::vector<T*> outputs;
    auto out_var_names = ctx.OutputNames(framework::GradVarName("X"));
    for (size_t j = 0; j < dx.size(); ++j) {
      if (out_var_names[j] != framework::kEmptyVarName &&
          dx[j]->numel() != 0UL) {
        T* ptr = dx[j]->mutable_data<T>(ctx.GetPlace());
        outputs.push_back(ptr);
      } else {
        outputs.push_back(nullptr);
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

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto tmp_out_data = memory::Alloc(dev_ctx, outputs.size() * sizeof(T*));
    memory::Copy(boost::get<platform::CUDAPlace>(dev_ctx.GetPlace()),
                 tmp_out_data->ptr(), platform::CPUPlace(),
                 reinterpret_cast<void*>(outputs.data()),
                 outputs.size() * sizeof(T*), dev_ctx.stream());

    int block_dims;
    int grid_dims;
    Get1DGpuLaunchConfig(dev_ctx, dy_pre * split_dim * dy_suf, &block_dims,
                         &grid_dims);
    unstack_kernel<T><<<grid_dims, block_dims, 0, dev_ctx.stream()>>>(
        dy_data, dy_pre, split_dim, dy_suf, split_dim,
        reinterpret_cast<T**>(tmp_out_data->ptr()));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    stack, ops::StackGPUKernel<plat::CUDADeviceContext, float>,
    ops::StackGPUKernel<plat::CUDADeviceContext, double>,
    ops::StackGPUKernel<plat::CUDADeviceContext, int>,
    ops::StackGPUKernel<plat::CUDADeviceContext, int64_t>,
    ops::StackGPUKernel<plat::CUDADeviceContext, plat::float16>);

REGISTER_OP_CUDA_KERNEL(
    stack_grad, ops::StackGradGPUKernel<plat::CUDADeviceContext, float>,
    ops::StackGradGPUKernel<plat::CUDADeviceContext, double>,
    ops::StackGradGPUKernel<plat::CUDADeviceContext, int>,
    ops::StackGradGPUKernel<plat::CUDADeviceContext, int64_t>,
    ops::StackGradGPUKernel<plat::CUDADeviceContext, plat::float16>);
