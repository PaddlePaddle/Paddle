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
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/fast_divmod.h"

namespace phi {

template <typename IndexT>
struct DivmodWarpper {
 public:
  void SetDivden(IndexT dividen) { divmoder = phi::funcs::FastDivMod(dividen); }
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

  void SetDivden(int64_t dividen) { dividen_ = dividen; }
  __device__ inline DivModT div_mod(int64_t val) {
    DivModT data;
    data[0] = val / dividen_;
    data[1] = val - data[0] * dividen_;
    return data;
  }

 private:
  int64_t dividen_;
};

constexpr int kWarpperSize = 64;
template <typename T, typename IndexT>
struct PointerArray : public DivmodWarpper<IndexT> {
 public:
  const T* data[kWarpperSize];
  PointerArray(const std::vector<const DenseTensor*>& x,
               int num,
               int64_t dividen) {
    this->SetDivden(dividen);
    for (auto i = 0; i < num; ++i) {
      data[i] = x[i]->data<T>();
    }
  }
};

template <typename Context, typename T, typename IndexT>
struct PointerToPointer : public DivmodWarpper<IndexT> {
 public:
  T** data;
  PointerToPointer(const Context& ctx,
                   const std::vector<const DenseTensor*>& x,
                   int num,
                   int64_t dividen) {
    this->SetDivden(dividen);
    auto byte_len = num * sizeof(T*);
    std::vector<const T*> x_datas(num);
    for (int i = 0; i < num; ++i) {
      x_datas[i] = x[i]->data<T>();
    }
    auto tmp_x_data = paddle::memory::Alloc(
        ctx.GetPlace(),
        byte_len,
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
    paddle::memory::Copy(ctx.GetPlace(),
                         tmp_x_data->ptr(),
                         phi::CPUPlace(),
                         reinterpret_cast<void*>(x_datas.data()),
                         x_datas.size() * sizeof(T*),
                         ctx.stream());
    data = reinterpret_cast<T**>(tmp_x_data->ptr());
  }
};

template <typename T, typename IndexT, typename WarpT>
__global__ void StackCUDAKernel(WarpT input_warpper,
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

template <typename T, typename Context>
void StackKernel(const Context& dev_ctx,
                 const std::vector<const DenseTensor*>& x,
                 int axis,
                 DenseTensor* out) {
  if (axis < 0) axis += (x[0]->dims().size() + 1);
  int n = static_cast<int>(x.size());
  T* y_data = dev_ctx.template Alloc<T>(out);

  // Split x dim from axis to matrix
  int64_t x_row = 1, x_col = 1;
  for (int i = 0; i < axis; ++i) {
    x_row *= x[0]->dims()[i];
  }
  x_col = x[0]->numel() / x_row;
  int64_t out_col = x_col * n;
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig2D(dev_ctx, out_col, x_row);

#define IMPL_STACK_CUDA_KERNEL(index_t, input_warpper)      \
  StackCUDAKernel<T, index_t, decltype(input_warpper)>      \
      <<<config.block_per_grid,                             \
         config.thread_per_block,                           \
         0,                                                 \
         dev_ctx.stream()>>>(input_warpper,                 \
                             static_cast<index_t>(x_col),   \
                             static_cast<index_t>(x_row),   \
                             static_cast<index_t>(out_col), \
                             y_data);

  bool use_int32 = out->numel() < std::numeric_limits<int32_t>::max();
  if (n <= kWarpperSize) {
    if (use_int32) {
      PointerArray<T, int32_t> ptr_array(x, n, x_col);
      IMPL_STACK_CUDA_KERNEL(int32_t, ptr_array);
    } else {
      PointerArray<T, int64_t> ptr_array(x, n, x_col);
      IMPL_STACK_CUDA_KERNEL(int64_t, ptr_array);
    }
  } else {
    if (use_int32) {
      PointerToPointer<Context, T, int32_t> ptr_array(dev_ctx, x, n, x_col);
      IMPL_STACK_CUDA_KERNEL(int32_t, ptr_array);
    } else {
      PointerToPointer<Context, T, int64_t> ptr_array(dev_ctx, x, n, x_col);
      IMPL_STACK_CUDA_KERNEL(int64_t, ptr_array);
    }
  }
#undef IMPL_STACK_CUDA_KERNEL
}
}  // namespace phi

PD_REGISTER_KERNEL(stack,
                   GPU,
                   ALL_LAYOUT,
                   phi::StackKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
