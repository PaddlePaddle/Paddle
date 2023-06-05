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

#include "paddle/phi/backends/gpu/gpu_context.h"
#ifndef PADDLE_WITH_XPU_KP
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#endif
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/errors.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void AddCudaFunctor(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    int axis,
                    DenseTensor* out) {
  std::vector<const DenseTensor*> inputs;
  inputs.reserve(2);
  std::vector<DenseTensor*> outputs;
  outputs.reserve(1);
  inputs.emplace_back(&x);
  inputs.emplace_back(&y);
  outputs.emplace_back(out);
  dev_ctx.template Alloc<T>(out);
  funcs::BroadcastKernel<T>(
      dev_ctx, inputs, &outputs, funcs::AddFunctor<T>(), axis);
}

template <int64_t VecSize>
__global__ void fused_add_cuda_forward(const float* x,
                                       const phi::bfloat16* y,
                                       float* out,
                                       int64_t numel) {
  int64_t i =
      (threadIdx.x + static_cast<int64_t>(blockIdx.x) * blockDim.x) * VecSize;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x * VecSize;

  for (; i + VecSize <= numel; i += stride) {
    phi::AlignedVector<float, VecSize> x_vec;
    phi::AlignedVector<phi::bfloat16, VecSize> y_vec;
    phi::AlignedVector<float, VecSize> out_vec;
    phi::Load(x + i, &x_vec);
    phi::Load(y + i, &y_vec);
#pragma unroll
    for (int j = 0; j < VecSize; ++j) {
      out_vec[j] = x_vec[j] + static_cast<float>(y_vec[j]);
    }
    phi::Store(out_vec, out + i);
  }

  for (; i < numel; ++i) {
    out[i] = x[i] + static_cast<float>(y[i]);
  }
}

template <typename Context>
void FusedAddCudaFunctor(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         DenseTensor* out) {
  int64_t numel = x.numel();
  auto place = phi::GPUPlace();
  int x_vec_size = phi::GetVectorizedSize(x.data<float>());
  int y_vec_size = phi::GetVectorizedSize(y.data<phi::bfloat16>());
  int out_vec_size = phi::GetVectorizedSize(out->data<float>());
  int vec_size = min(x_vec_size, min(y_vec_size, out_vec_size));
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel, vec_size);
  auto threads = config.GetBlockSize();
  auto blocks = config.block_per_grid;
  auto stream = dev_ctx.stream();
  switch (vec_size) {
    case 4:
      fused_add_cuda_forward<4><<<blocks, threads, 0, stream>>>(
          x.data<float>(), y.data<phi::bfloat16>(), out->data<float>(), numel);
      break;

    case 2:
      fused_add_cuda_forward<2><<<blocks, threads, 0, stream>>>(
          x.data<float>(), y.data<phi::bfloat16>(), out->data<float>(), numel);
      break;

    case 1:
      fused_add_cuda_forward<1><<<blocks, threads, 0, stream>>>(
          x.data<float>(), y.data<phi::bfloat16>(), out->data<float>(), numel);
      break;
    default: {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupported vectorized size: %d !", vec_size));
      break;
    }
  }
}

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out) {
  AddCudaFunctor<T, Context>(dev_ctx, x, y, -1, out);
  if (x.dtype() == y.dtype()) {
    AddCudaFunctor<T, Context>(dev_ctx, x, y, -1, out);
  } else {
    VLOG(2) << "x dtype:" << x.dtype() << " != y dtype:" << y.dtype();
    PADDLE_ENFORCE_EQ(
        x.dtype(),
        phi::DataType::FLOAT32,
        phi::errors::InvalidArgument("The x should be float32 dtype in x+y."));
    PADDLE_ENFORCE_EQ(
        y.dtype(),
        phi::DataType::BFLOAT16,
        phi::errors::InvalidArgument("The y should be bfloat16 dtype in x+y."));
    FusedAddCudaFunctor<Context>(dev_ctx, x, y, out);
  }
}

template <typename T, typename Context>
void GradAddKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  AddCudaFunctor<T>(dev_ctx, x, y, -1, out);
}

}  // namespace phi

#ifdef PADDLE_WITH_XPU_KP
PD_REGISTER_KERNEL(add, KPS, ALL_LAYOUT, phi::AddKernel, float) {}
#else

using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;
using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(add,
                   KPS,
                   ALL_LAYOUT,
                   phi::AddKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   complex64,
                   complex128) {}

PD_REGISTER_KERNEL(grad_add,
                   KPS,
                   ALL_LAYOUT,
                   phi::GradAddKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   complex64,
                   complex128) {}
#endif
