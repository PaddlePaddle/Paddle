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
  __host__ void SetDivden(IndexT dividen) {
    divmoder = phi::funcs::FastDivMod(dividen);
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

  __host__ void SetDivden(int64_t dividen) { dividen_ = dividen; }
  __device__ inline DivModT div_mod(int64_t val) {
    DivModT data;
    data[0] = val / dividen_;
    data[1] = val - data[0] * dividen_;
    return data;
  }

 private:
  int64_t dividen_;
};

constexpr int kWarpperSize = 256;
template <typename T, typename IndexT, bool IsDataWarpperd>
struct DataWarpper : public DivmodWarpper<IndexT> {
  const T* data[kWarpperSize];
};

template <typename T, typename IndexT>
struct DataWarpper<T, IndexT, false> : public DivmodWarpper<IndexT> {
  T** data;
};

template <typename Context, typename T>
T** PackDataAndTransfer(const Context& dev_ctx,
                        const std::vector<const DenseTensor*>& x,
                        int num) {
  std::vector<const T*> x_datas(num);
  for (int i = 0; i < num; ++i) {
    x_datas[i] = x[i]->data<T>();
  }
  auto byte_len = num * sizeof(T*);
  auto tmp_x_data = paddle::memory::Alloc(
      dev_ctx.GetPlace(),
      byte_len,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  paddle::memory::Copy(dev_ctx.GetPlace(),
                       tmp_x_data->ptr(),
                       phi::CPUPlace(),
                       reinterpret_cast<void*>(x_datas.data()),
                       byte_len,
                       dev_ctx.stream());
  return reinterpret_cast<T**>(tmp_x_data->ptr());
}

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

  if (out->numel() < std::numeric_limits<int32_t>::max()) {
    if (n <= kWarpperSize) {
      DataWarpper<T, int32_t, true> data_warpper;
      for (auto i = 0; i < n; ++i) {
        data_warpper.data[i] = x[i]->data<T>();
      }
      data_warpper.SetDivden(x_col);
      IMPL_STACK_CUDA_KERNEL(int32_t, data_warpper);
    } else {
      DataWarpper<T, int32_t, false> data_warpper;
      T** pack_ptr = PackDataAndTransfer<Context, T>(dev_ctx, x, n);
      data_warpper.data = pack_ptr;
      data_warpper.SetDivden(x_col);
      IMPL_STACK_CUDA_KERNEL(int32_t, data_warpper);
    }
  } else {
    if (n <= kWarpperSize) {
      DataWarpper<T, int64_t, true> data_warpper;
      for (auto i = 0; i < n; ++i) {
        data_warpper.data[i] = x[i]->data<T>();
      }
      data_warpper.SetDivden(x_col);
      IMPL_STACK_CUDA_KERNEL(int64_t, data_warpper);
    } else {
      DataWarpper<T, int64_t, false> data_warpper;
      T** pack_ptr = PackDataAndTransfer<Context, T>(dev_ctx, x, n);
      data_warpper.data = pack_ptr;
      data_warpper.SetDivden(x_col);
      IMPL_STACK_CUDA_KERNEL(int64_t, data_warpper);
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
