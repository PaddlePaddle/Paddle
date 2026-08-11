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

#include "paddle/common/enforce.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename T, int N>
struct alignas(16) VectorType {
  T data[N];
};

template <typename ScaleT, bool using_ue8m0_scale>
__device__ __forceinline__ float LoadScale(const ScaleT* ptr, int64_t idx) {
  if constexpr (using_ue8m0_scale) {
    int packed_scale = reinterpret_cast<const int*>(ptr)[idx / 4];
    int scale_offset = idx % 4;
    uint8_t scale_u8 = (packed_scale >> (scale_offset * 8)) & 0xFF;
    int val_as_int = static_cast<int>(scale_u8) << 23;
    return __int_as_float(val_as_int);
  } else {
    return ptr[idx];
  }
}

template <typename ScaleT, bool using_ue8m0_scale>
__global__ void FusedActDequant(const phi::float8_e4m3fn* __restrict__ Xin,
                                const ScaleT* __restrict__ Xscale,
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
    float this_scale = LoadScale<ScaleT, using_ue8m0_scale>(Xscale, scale_idx);

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

      int64_t scale_idx =
          (int64_t)this_row_idx * (int64_t)Xscale_stride + (x_offset / 128);
      float this_scale =
          LoadScale<ScaleT, using_ue8m0_scale>(Xscale, scale_idx);
      X_value *= this_scale;
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
  PADDLE_ENFORCE_LE_INT_MAX(x_dims[0], "fused_act_dequant rows");
  PADDLE_ENFORCE_LE_INT_MAX(x_dims[1], "fused_act_dequant cols");
  PADDLE_ENFORCE_LE_UINT32_MAX(x_dims[0], "fused_act_dequant grid.x");
  int rows = static_cast<int>(x_dims[0]);
  int cols = static_cast<int>(x_dims[1]);

  out->Resize({rows, cols});
  dev_ctx.template Alloc<phi::bfloat16>(out);

  auto out_ptr = reinterpret_cast<void*>(out->template data<phi::bfloat16>());

  dim3 grid(static_cast<uint32_t>(x_dims[0]));
  dim3 block(256);

  if (x_scale.dtype() == phi::DataType::FLOAT32) {
    FusedActDequant<float, false>
        <<<grid, block, 0, dev_ctx.stream()>>>(x.data<phi::float8_e4m3fn>(),
                                               x_scale.data<float>(),
                                               out->data<phi::bfloat16>(),
                                               rows,
                                               cols);
  } else if (x_scale.dtype() == phi::DataType::INT32) {
    FusedActDequant<int, true>
        <<<grid, block, 0, dev_ctx.stream()>>>(x.data<phi::float8_e4m3fn>(),
                                               x_scale.data<int>(),
                                               out->data<phi::bfloat16>(),
                                               rows,
                                               cols);
  }

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
                   phi::float8_e4m3fn) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BFLOAT16);
}
