// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename T, int N>
struct alignas(16) VectorType {
  T data[N];
};

__global__ void FusedActDequant(const phi::float8_e4m3fn* __restrict__ Xin,
                                const float* __restrict__ Xscale,
                                phi::bfloat16* __restrict__ out,
                                const int rows,
                                const int cols) {
  const int this_row_idx = blockIdx.x;
  if (this_row_idx >= rows) return;

  const int Xscale_stride = (cols + 127) / 128;

  const int vector_size = 16;

  const int num_vectors = cols / vector_size;
  const int remaining_elements = cols % vector_size;

  const int tid = threadIdx.x;

  for (int vec_idx = tid; vec_idx < num_vectors; vec_idx += blockDim.x) {
    int x_offset = vec_idx * vector_size;
    int64_t X_idx = (int64_t)this_row_idx * (int64_t)cols + (int64_t)x_offset;

    const VectorType<__nv_fp8_e4m3, vector_size>* X_vec_ptr =
        reinterpret_cast<const VectorType<__nv_fp8_e4m3, vector_size>*>(Xin +
                                                                        X_idx);
    VectorType<__nv_fp8_e4m3, vector_size> X_vec = X_vec_ptr[0];

    int64_t scale_idx =
        (int64_t)this_row_idx * (int64_t)Xscale_stride + (x_offset / 128);
    float this_scale = Xscale[scale_idx];

    VectorType<__nv_bfloat16, vector_size> out_vec;

#pragma unroll
    for (int i = 0; i < vector_size; ++i) {
      float X_value = static_cast<float>(X_vec.data[i]);
      X_value *= this_scale;
      out_vec.data[i] = __float2bfloat16(X_value);
    }

    VectorType<__nv_bfloat16, vector_size>* out_vec_ptr =
        reinterpret_cast<VectorType<__nv_bfloat16, vector_size>*>(out + X_idx);
    out_vec_ptr[0] = out_vec;
  }

  if (remaining_elements > 0) {
    int x_offset = num_vectors * vector_size;
    int64_t X_idx = (int64_t)this_row_idx * (int64_t)cols + (int64_t)x_offset;
    int64_t idx = X_idx + tid;
    if (tid < remaining_elements) {
      float X_value = static_cast<float>(Xin[idx]);
      X_value *= Xscale[(int64_t)this_row_idx * (int64_t)Xscale_stride +
                        (x_offset / 128)];
      out[idx] = __float2bfloat16(X_value);
    }
  }
}

template <typename T, typename Context>
void FusedActDequantKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& x_scale,
                           DenseTensor* out) {
  auto x_dims = x.dims();
  int rows = x_dims[0];
  int cols = x_dims[1];

  out->Resize({rows, cols});
  dev_ctx.template Alloc<phi::dtype::bfloat16>(out);

  auto out_ptr =
      reinterpret_cast<void*>(out->template data<phi::dtype::bfloat16>());
  cudaMemsetAsync(
      out_ptr, 0, sizeof(phi::dtype::bfloat16) * rows * cols, dev_ctx.stream());

  dim3 grid(rows);
  dim3 block(256);

  FusedActDequant<<<grid, block, 0, dev_ctx.stream()>>>(
      x.data<phi::dtype::float8_e4m3fn>(),
      x_scale.data<float>(),
      out->data<phi::dtype::bfloat16>(),
      rows,
      cols);

#ifdef PADDLE_WITH_CUDA_CHECK
  auto cuda_error = cudaGetLastError();
  PADDLE_ENFORCE_GPU_SUCCESS(cuda_error);
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(fused_act_dequant,
                   GPU,
                   ALL_LAYOUT,
                   phi::FusedActDequantKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float8_e4m3fn) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BFLOAT16);
}
