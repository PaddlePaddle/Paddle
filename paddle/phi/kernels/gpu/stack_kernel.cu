// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/stack_kernel.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/fast_divmod.h"

namespace phi {

template <typename IndexT>
struct DivmodWarpper {
 public:
  void SetDivisor(IndexT divisor) {
    divmoder = phi::funcs::FastDivMod(divisor);
  }
  __device__ inline phi::funcs::FastDivMod::DivModT div_mod(IndexT val) {
    return divmoder.Divmod(val);
  }

 private:
  phi::funcs::FastDivMod divmoder;
};

template <>
struct DivmodWarpper<int64_t> {
 public:
  using DivModT = phi::AlignedVector<int64_t, 2>;

  void SetDivisor(int64_t divisor) { dividen_ = divisor; }
  __device__ inline DivModT div_mod(int64_t val) {
    DivModT data;
    data[0] = val / dividen_;
    data[1] = val - data[0] * dividen_;
    return data;
  }

 private:
  int64_t dividen_;
};

template <typename T, typename IndexT, int Size>
struct PointerArray : public DivmodWarpper<IndexT> {
 public:
  const T* data[Size];
  PointerArray(const std::vector<const DenseTensor*>& x,
               int num,
               IndexT divisor) {
    this->SetDivisor(divisor);
    for (auto i = 0; i < num; ++i) {
      data[i] = x[i]->data<T>();
    }
  }
};

template <typename Context, typename T, typename IndexT>
struct PointerToPointer : public DivmodWarpper<IndexT> {
 public:
  T** data{nullptr};
  PointerToPointer(const Context& ctx,
                   const std::vector<const DenseTensor*>& x,
                   IndexT num,
                   IndexT divisor,
                   paddle::memory::AllocationPtr* dev_ins_ptr) {
    this->SetDivisor(divisor);
    std::vector<const T*> x_datas(num);
    for (int i = 0; i < num; ++i) {
      x_datas[i] = x[i]->data<T>();
    }
    *dev_ins_ptr = paddle::memory::Alloc(
        ctx.GetPlace(),
        num * sizeof(T*),
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
    paddle::memory::Copy(ctx.GetPlace(),
                         (*dev_ins_ptr)->ptr(),
                         phi::CPUPlace(),
                         reinterpret_cast<void*>(x_datas.data()),
                         num * sizeof(T*),
                         ctx.stream());
    data = reinterpret_cast<T**>((*dev_ins_ptr)->ptr());
  }
};

template <typename T, typename IndexT, typename WrapT>
__global__ void StackCUDAKernel(WrapT input_warpper,
                                IndexT split_size,
                                IndexT rows,
                                IndexT cols,
                                T* __restrict__ output) {
  IndexT grid_x = static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x;
  IndexT grid_x_stride = static_cast<IndexT>(blockDim.x) * gridDim.x;
  IndexT grid_y_stride = static_cast<IndexT>(blockDim.y) * gridDim.y;

  for (; grid_x < cols; grid_x += grid_x_stride) {
    IndexT grid_y = static_cast<IndexT>(blockIdx.y) * blockDim.y + threadIdx.y;

    auto divmod_rslt = input_warpper.div_mod(grid_x);
    const T* input_ptr = input_warpper.data[divmod_rslt[0]];
#pragma unroll
    for (; grid_y < rows; grid_y += grid_y_stride) {
      output[grid_y * cols + grid_x] =
          input_ptr[grid_y * split_size + divmod_rslt[1]];
    }
  }
}

template <typename T, typename IndexT, typename Context>
void LaunchStackCUDAKernelWithIndexType(
    const Context& ctx,
    const IndexT x_col,
    const IndexT x_row,
    const IndexT out_col,
    const phi::backends::gpu::GpuLaunchConfig& cfg,
    const std::vector<const DenseTensor*>& x,
    T* dst_data) {
  int num = static_cast<int>(x.size());
#define IMPL_STACK_CUDA_KERNEL_CASE(size_, ...)              \
  case size_: {                                              \
    PointerArray<T, IndexT, size_> ptr_array(x, num, x_col); \
    __VA_ARGS__;                                             \
  } break;

#define IMPL_STACK_CUDA_KERNEL_HELPER(...)        \
  IMPL_STACK_CUDA_KERNEL_CASE(4, ##__VA_ARGS__);  \
  IMPL_STACK_CUDA_KERNEL_CASE(8, ##__VA_ARGS__);  \
  IMPL_STACK_CUDA_KERNEL_CASE(16, ##__VA_ARGS__); \
  IMPL_STACK_CUDA_KERNEL_CASE(32, ##__VA_ARGS__); \
  IMPL_STACK_CUDA_KERNEL_CASE(64, ##__VA_ARGS__); \
  IMPL_STACK_CUDA_KERNEL_CASE(128, ##__VA_ARGS__);

  switch (phi::backends::gpu::RoundToNextHighPowOfTwo(num, 4)) {
    IMPL_STACK_CUDA_KERNEL_HELPER(
        StackCUDAKernel<T, IndexT, decltype(ptr_array)>
        <<<cfg.block_per_grid, cfg.thread_per_block, 0, ctx.stream()>>>(
            ptr_array, x_col, x_row, out_col, dst_data));
    default: {
      paddle::memory::AllocationPtr dev_ins_ptr{nullptr};
      PointerToPointer<Context, T, IndexT> ptr_array(
          ctx, x, num, x_col, &dev_ins_ptr);
      StackCUDAKernel<T, IndexT, decltype(ptr_array)>
          <<<cfg.block_per_grid, cfg.thread_per_block, 0, ctx.stream()>>>(
              ptr_array, x_col, x_row, out_col, dst_data);
    }
  }
#undef IMPL_STACK_CUDA_KERNEL_HELPER
#undef IMPL_STACK_CUDA_KERNEL_CASE
}

template <typename T, typename Context>
void StackKernel(const Context& dev_ctx,
                 const std::vector<const DenseTensor*>& x,
                 int axis,
                 DenseTensor* out) {
  if (axis < 0) axis += (x[0]->dims().size() + 1);
  int num = static_cast<int>(x.size());
  T* dst_data = dev_ctx.template Alloc<T>(out);

  // Split x dim from axis to matrix
  int64_t x_row = 1, x_col = 1;
  for (int i = 0; i < axis; ++i) {
    x_row *= x[0]->dims()[i];
  }
  x_col = x[0]->numel() / x_row;
  int64_t out_col = x_col * num;
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig2D(dev_ctx, out_col, x_row);

  if (out->numel() < std::numeric_limits<int32_t>::max()) {
    LaunchStackCUDAKernelWithIndexType<T, int32_t, Context>(
        dev_ctx, x_col, x_row, out_col, config, x, dst_data);
  } else {
    LaunchStackCUDAKernelWithIndexType<T, int64_t, Context>(
        dev_ctx, x_col, x_row, out_col, config, x, dst_data);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(stack,
                   GPU,
                   ALL_LAYOUT,
                   phi::StackKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
