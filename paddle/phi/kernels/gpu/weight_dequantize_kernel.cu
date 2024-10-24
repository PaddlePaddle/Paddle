/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/weight_dequantize_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/transpose_kernel.h"

#if defined(PADDLE_WITH_CUTLASS)
#include "paddle/phi/kernels/funcs/weight_dequant_functor.h"
#endif

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#endif

namespace phi {

#ifdef PADDLE_WITH_HIP

#define NUMPERTHREAD 16
template <typename T, int Size>
struct alignas(sizeof(T) * Size) aligned_vector {
  T val[Size];
};
using int8_8 = aligned_vector<int8_t, 8>;

template <typename T>
__device__ inline aligned_vector<T, 8> i82h_convert8(int8_8 signed_chars,
                                                     T scale) {
  aligned_vector<T, 8> halves;
  halves.val[0] = static_cast<T>(static_cast<float>(signed_chars.val[0]) *
                                 static_cast<float>(scale));
  halves.val[1] = static_cast<T>(static_cast<float>(signed_chars.val[1]) *
                                 static_cast<float>(scale));
  halves.val[2] = static_cast<T>(static_cast<float>(signed_chars.val[2]) *
                                 static_cast<float>(scale));
  halves.val[3] = static_cast<T>(static_cast<float>(signed_chars.val[3]) *
                                 static_cast<float>(scale));
  halves.val[4] = static_cast<T>(static_cast<float>(signed_chars.val[4]) *
                                 static_cast<float>(scale));
  halves.val[5] = static_cast<T>(static_cast<float>(signed_chars.val[5]) *
                                 static_cast<float>(scale));
  halves.val[6] = static_cast<T>(static_cast<float>(signed_chars.val[6]) *
                                 static_cast<float>(scale));
  halves.val[7] = static_cast<T>(static_cast<float>(signed_chars.val[7]) *
                                 static_cast<float>(scale));
  return halves;
}

template <>
__device__ inline aligned_vector<half, 8> i82h_convert8(int8_8 signed_chars,
                                                        half scale) {
  aligned_vector<half, 8> halves;
  halves.val[0] = __float2half(static_cast<float>(signed_chars.val[0]) *
                               __half2float(scale));
  halves.val[1] = __float2half(static_cast<float>(signed_chars.val[1]) *
                               __half2float(scale));
  halves.val[2] = __float2half(static_cast<float>(signed_chars.val[2]) *
                               __half2float(scale));
  halves.val[3] = __float2half(static_cast<float>(signed_chars.val[3]) *
                               __half2float(scale));
  halves.val[4] = __float2half(static_cast<float>(signed_chars.val[4]) *
                               __half2float(scale));
  halves.val[5] = __float2half(static_cast<float>(signed_chars.val[5]) *
                               __half2float(scale));
  halves.val[6] = __float2half(static_cast<float>(signed_chars.val[6]) *
                               __half2float(scale));
  halves.val[7] = __float2half(static_cast<float>(signed_chars.val[7]) *
                               __half2float(scale));
  return halves;
}

struct uint4_2 {
  uint8_t data;

  explicit uint4_2(uint8_t x = 0, uint8_t y = 0) {
    setX(x);
    setY(y);
  }

  __host__ __device__ uint8_t getX() const {
    return data & 0x0F;  // lower 4 bits
  }

  __host__ __device__ uint8_t getY() const {
    return (data >> 4) & 0x0F;  // upper 4 bits
  }

  __host__ __device__ void setX(uint8_t x) {
    data = (data & 0xF0) | (x & 0x0F);  // set the lower 4 bits
  }

  __host__ __device__ void setY(uint8_t y) {
    data = (data & 0x0F) | ((y & 0x0F) << 4);  // set the upper 4 bits
  }
};
using uint4_2_8 = aligned_vector<uint4_2, 8>;

template <typename T>
__device__ inline aligned_vector<aligned_vector<T, 8>, 2> i42h_convert8_2(
    uint4_2_8 signed_chars, T scale_0, T scale_1) {
  aligned_vector<aligned_vector<T, 8>, 2> halves;
  aligned_vector<T, 8> halves_0;
  aligned_vector<T, 8> halves_1;
  halves_0.val[0] = static_cast<T>(
      static_cast<float>((int8_t)signed_chars.val[0].getX() - 8) *
      static_cast<float>(scale_0));
  halves_0.val[1] = static_cast<T>(
      static_cast<float>((int8_t)signed_chars.val[1].getX() - 8) *
      static_cast<float>(scale_0));
  halves_0.val[2] = static_cast<T>(
      static_cast<float>((int8_t)signed_chars.val[2].getX() - 8) *
      static_cast<float>(scale_0));
  halves_0.val[3] = static_cast<T>(
      static_cast<float>((int8_t)signed_chars.val[3].getX() - 8) *
      static_cast<float>(scale_0));
  halves_0.val[4] = static_cast<T>(
      static_cast<float>((int8_t)signed_chars.val[4].getX() - 8) *
      static_cast<float>(scale_0));
  halves_0.val[5] = static_cast<T>(
      static_cast<float>((int8_t)signed_chars.val[5].getX() - 8) *
      static_cast<float>(scale_0));
  halves_0.val[6] = static_cast<T>(
      static_cast<float>((int8_t)signed_chars.val[6].getX() - 8) *
      static_cast<float>(scale_0));
  halves_0.val[7] = static_cast<T>(
      static_cast<float>((int8_t)signed_chars.val[7].getX() - 8) *
      static_cast<float>(scale_0));
  halves_1.val[0] = static_cast<T>(
      static_cast<float>((int8_t)signed_chars.val[0].getY() - 8) *
      static_cast<float>(scale_1));
  halves_1.val[1] = static_cast<T>(
      static_cast<float>((int8_t)signed_chars.val[1].getY() - 8) *
      static_cast<float>(scale_1));
  halves_1.val[2] = static_cast<T>(
      static_cast<float>((int8_t)signed_chars.val[2].getY() - 8) *
      static_cast<float>(scale_1));
  halves_1.val[3] = static_cast<T>(
      static_cast<float>((int8_t)signed_chars.val[3].getY() - 8) *
      static_cast<float>(scale_1));
  halves_1.val[4] = static_cast<T>(
      static_cast<float>((int8_t)signed_chars.val[4].getY() - 8) *
      static_cast<float>(scale_1));
  halves_1.val[5] = static_cast<T>(
      static_cast<float>((int8_t)signed_chars.val[5].getY() - 8) *
      static_cast<float>(scale_1));
  halves_1.val[6] = static_cast<T>(
      static_cast<float>((int8_t)signed_chars.val[6].getY() - 8) *
      static_cast<float>(scale_1));
  halves_1.val[7] = static_cast<T>(
      static_cast<float>((int8_t)signed_chars.val[7].getY() - 8) *
      static_cast<float>(scale_1));
  halves.val[0] = halves_0;
  halves.val[1] = halves_1;
  return halves;
}

template <>
__device__ inline aligned_vector<aligned_vector<half, 8>, 2> i42h_convert8_2(
    uint4_2_8 signed_chars, half scale_0, half scale_1) {
  aligned_vector<aligned_vector<half, 8>, 2> halves;
  aligned_vector<half, 8> halves_0;
  aligned_vector<half, 8> halves_1;
  halves_0.val[0] =
      __float2half(static_cast<float>((int8_t)signed_chars.val[0].getX() - 8) *
                   __half2float(scale_0));
  halves_0.val[1] =
      __float2half(static_cast<float>((int8_t)signed_chars.val[1].getX() - 8) *
                   __half2float(scale_0));
  halves_0.val[2] =
      __float2half(static_cast<float>((int8_t)signed_chars.val[2].getX() - 8) *
                   __half2float(scale_0));
  halves_0.val[3] =
      __float2half(static_cast<float>((int8_t)signed_chars.val[3].getX() - 8) *
                   __half2float(scale_0));
  halves_0.val[4] =
      __float2half(static_cast<float>((int8_t)signed_chars.val[4].getX() - 8) *
                   __half2float(scale_0));
  halves_0.val[5] =
      __float2half(static_cast<float>((int8_t)signed_chars.val[5].getX() - 8) *
                   __half2float(scale_0));
  halves_0.val[6] =
      __float2half(static_cast<float>((int8_t)signed_chars.val[6].getX() - 8) *
                   __half2float(scale_0));
  halves_0.val[7] =
      __float2half(static_cast<float>((int8_t)signed_chars.val[7].getX() - 8) *
                   __half2float(scale_0));
  halves_1.val[0] =
      __float2half(static_cast<float>((int8_t)signed_chars.val[0].getY() - 8) *
                   __half2float(scale_1));
  halves_1.val[1] =
      __float2half(static_cast<float>((int8_t)signed_chars.val[1].getY() - 8) *
                   __half2float(scale_1));
  halves_1.val[2] =
      __float2half(static_cast<float>((int8_t)signed_chars.val[2].getY() - 8) *
                   __half2float(scale_1));
  halves_1.val[3] =
      __float2half(static_cast<float>((int8_t)signed_chars.val[3].getY() - 8) *
                   __half2float(scale_1));
  halves_1.val[4] =
      __float2half(static_cast<float>((int8_t)signed_chars.val[4].getY() - 8) *
                   __half2float(scale_1));
  halves_1.val[5] =
      __float2half(static_cast<float>((int8_t)signed_chars.val[5].getY() - 8) *
                   __half2float(scale_1));
  halves_1.val[6] =
      __float2half(static_cast<float>((int8_t)signed_chars.val[6].getY() - 8) *
                   __half2float(scale_1));
  halves_1.val[7] =
      __float2half(static_cast<float>((int8_t)signed_chars.val[7].getY() - 8) *
                   __half2float(scale_1));
  halves.val[0] = halves_0;
  halves.val[1] = halves_1;
  return halves;
}

template <typename T>
__global__ void int8_weight_only_dequant(int8_t* mat,
                                         T* scales,
                                         T* mat_res,
                                         unsigned int k,
                                         unsigned int k_iteration) {
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  int8_8* mat8 = reinterpret_cast<int8_8*>(mat);
  aligned_vector<T, 8>* mat_res8 =
      reinterpret_cast<aligned_vector<T, 8>*>(mat_res);
  T scale = scales[row];

#pragma unroll
  for (unsigned int iteration = 0; iteration < k_iteration; iteration++) {
    unsigned int gidx = tid + iteration * blockDim.x;
    unsigned int gdatax = NUMPERTHREAD / 8 * gidx;

#pragma unroll
    for (unsigned int it = 0; it < NUMPERTHREAD / 8 / 2; it++) {
      if (gdatax + 2 * it + 1 < k / 8) {
        mat_res8[row * (k / 8) + gdatax + 2 * it] =
            i82h_convert8(mat8[row * (k / 8) + gdatax + 2 * it], scale);
        mat_res8[row * (k / 8) + gdatax + 2 * it + 1] =
            i82h_convert8(mat8[row * (k / 8) + gdatax + 2 * it + 1], scale);
      }
    }
  }
}

template <typename T>
__global__ void int8_weight_only_dequant(int8_t* mat,
                                         T* scales,
                                         T* mat_res,
                                         unsigned int k,
                                         unsigned int groupsize,
                                         unsigned int k_iteration) {
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  int8_8* mat8 = reinterpret_cast<int8_8*>(mat);
  aligned_vector<T, 8>* mat_res8 =
      reinterpret_cast<aligned_vector<T, 8>*>(mat_res);

#pragma unroll
  for (unsigned int iteration = 0; iteration < k_iteration; iteration++) {
    unsigned int gidx = tid + iteration * blockDim.x;
    unsigned int gdatax = NUMPERTHREAD / 8 * gidx;

    T scale = scales[row * (k / groupsize) + gidx * NUMPERTHREAD / groupsize];

#pragma unroll
    for (unsigned int it = 0; it < NUMPERTHREAD / 8 / 2; it++) {
      if (gdatax + 2 * it + 1 < k / 8) {
        mat_res8[row * (k / 8) + gdatax + 2 * it] =
            i82h_convert8(mat8[row * (k / 8) + gdatax + 2 * it], scale);
        mat_res8[row * (k / 8) + gdatax + 2 * it + 1] =
            i82h_convert8(mat8[row * (k / 8) + gdatax + 2 * it + 1], scale);
      }
    }
  }
}

template <typename T>
__global__ void int4_weight_only_dequant(uint4_2* mat,
                                         T* scales,
                                         T* mat_res,
                                         unsigned int k,
                                         unsigned int k_iteration) {
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  uint4_2_8* mat16 = reinterpret_cast<uint4_2_8*>(mat);
  aligned_vector<T, 8>* mat_res8 =
      reinterpret_cast<aligned_vector<T, 8>*>(mat_res);
  aligned_vector<aligned_vector<T, 8>, 2> mat_res16;

  T scale_0 = scales[2 * row];
  T scale_1 = scales[2 * row + 1];

#pragma unroll
  for (unsigned int iteration = 0; iteration < k_iteration; iteration++) {
    unsigned int gidx = tid + iteration * blockDim.x;
    unsigned int gdatax = NUMPERTHREAD / 8 * gidx;

#pragma unroll
    for (unsigned int it = 0; it < NUMPERTHREAD / 8 / 2; it++) {
      if (gdatax + 2 * it + 1 < k / 8) {
        mat_res16 = i42h_convert8_2(
            mat16[row * (k / 8) + gdatax + 2 * it], scale_0, scale_1);
        mat_res8[2 * row * (k / 8) + gdatax + 2 * it] = mat_res16.val[0];
        mat_res8[(2 * row + 1) * (k / 8) + gdatax + 2 * it] = mat_res16.val[1];
        mat_res16 = i42h_convert8_2(
            mat16[row * (k / 8) + gdatax + 2 * it + 1], scale_0, scale_1);
        mat_res8[2 * row * (k / 8) + gdatax + 2 * it + 1] = mat_res16.val[0];
        mat_res8[(2 * row + 1) * (k / 8) + gdatax + 2 * it + 1] =
            mat_res16.val[1];
      }
    }
  }
}

template <typename T>
__global__ void int4_weight_only_dequant(uint4_2* mat,
                                         T* scales,
                                         T* mat_res,
                                         unsigned int k,
                                         unsigned int groupsize,
                                         unsigned int k_iteration) {
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  uint4_2_8* mat16 = reinterpret_cast<uint4_2_8*>(mat);
  aligned_vector<T, 8>* mat_res8 =
      reinterpret_cast<aligned_vector<T, 8>*>(mat_res);
  aligned_vector<aligned_vector<T, 8>, 2> mat_res16;

#pragma unroll
  for (unsigned int iteration = 0; iteration < k_iteration; iteration++) {
    unsigned int gidx = tid + iteration * blockDim.x;
    unsigned int gdatax = NUMPERTHREAD / 8 * gidx;

    T scale_0 =
        scales[2 * row * (k / groupsize) + gidx * NUMPERTHREAD / groupsize];
    T scale_1 = scales[(2 * row + 1) * (k / groupsize) +
                       gidx * NUMPERTHREAD / groupsize];

#pragma unroll
    for (unsigned int it = 0; it < NUMPERTHREAD / 8 / 2; it++) {
      if (gdatax + 2 * it + 1 < k / 8) {
        mat_res16 = i42h_convert8_2(
            mat16[row * (k / 8) + gdatax + 2 * it], scale_0, scale_1);
        mat_res8[2 * row * (k / 8) + gdatax + 2 * it] = mat_res16.val[0];
        mat_res8[(2 * row + 1) * (k / 8) + gdatax + 2 * it] = mat_res16.val[1];
        mat_res16 = i42h_convert8_2(
            mat16[row * (k / 8) + gdatax + 2 * it + 1], scale_0, scale_1);
        mat_res8[2 * row * (k / 8) + gdatax + 2 * it + 1] = mat_res16.val[0];
        mat_res8[(2 * row + 1) * (k / 8) + gdatax + 2 * it + 1] =
            mat_res16.val[1];
      }
    }
  }
}

template <typename T, typename Context>
void WeightDequantize(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& scale,
                      const std::string& algo,
                      const int32_t group_size,
                      DenseTensor* out) {
  using DataType = typename PDDataTypeTraits<T>::DataType;
  int n = scale.dims()[0];
  int k = x.dims()[1];
  PADDLE_ENFORCE_EQ(
      (k % NUMPERTHREAD == 0),
      true,
      common::errors::InvalidArgument(
          "Currently, WeightDequantize only support k % NUMPERTHREAD == 0."));
  unsigned int block_dim_x = 256;
  unsigned int kperblock = block_dim_x * NUMPERTHREAD;
  unsigned int block_dim_y = 1;
  unsigned int k_iteration =
      k % kperblock == 0 ? k / kperblock : k / kperblock + 1;
  dim3 grid(1, n / block_dim_y);
  dim3 block(block_dim_x, block_dim_y);
  auto stream = dev_ctx.stream();

  if (algo == "weight_only_int8" && group_size == -1) {
    int8_weight_only_dequant<DataType><<<grid, block, 0, stream>>>(
        const_cast<int8_t*>(x.data<int8_t>()),
        const_cast<DataType*>(
            reinterpret_cast<const DataType*>(scale.data<T>())),
        reinterpret_cast<DataType*>(out->data<T>()),
        k,
        k_iteration);
  } else if (algo == "weight_only_int8" && group_size > 0) {
    int8_weight_only_dequant<DataType><<<grid, block, 0, stream>>>(
        const_cast<int8_t*>(x.data<int8_t>()),
        const_cast<DataType*>(
            reinterpret_cast<const DataType*>(scale.data<T>())),
        reinterpret_cast<DataType*>(out->data<T>()),
        k,
        group_size,
        k_iteration);
  } else if (algo == "weight_only_int4" && group_size == -1) {
    grid.y /= 2;
    int4_weight_only_dequant<DataType><<<grid, block, 0, stream>>>(
        reinterpret_cast<uint4_2*>(const_cast<int8_t*>(x.data<int8_t>())),
        const_cast<DataType*>(
            reinterpret_cast<const DataType*>(scale.data<T>())),
        reinterpret_cast<DataType*>(out->data<T>()),
        k,
        k_iteration);
  } else if (algo == "weight_only_int4" && group_size > 0) {
    grid.y /= 2;
    int4_weight_only_dequant<DataType><<<grid, block, 0, stream>>>(
        reinterpret_cast<uint4_2*>(const_cast<int8_t*>(x.data<int8_t>())),
        const_cast<DataType*>(
            reinterpret_cast<const DataType*>(scale.data<T>())),
        reinterpret_cast<DataType*>(out->data<T>()),
        k,
        group_size,
        k_iteration);
  }
}

#endif

template <typename T, typename Context>
void WeightDequantizeKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& scale,
                            const std::string& algo,
                            DataType out_dtype,
                            int32_t group_size,
                            DenseTensor* out) {
#if defined(PADDLE_WITH_CUTLASS)
  auto out_dims = out->dims();
  dev_ctx.template Alloc<T>(out);
  WeightDequantize<T, Context>(dev_ctx, x, scale, algo, true, group_size, out);
  out->Resize({{out_dims[1], out_dims[0]}});
  auto out_tmp = Transpose<T, Context>(dev_ctx, *out, {1, 0});
  out->ShareDataWith(out_tmp);
#elif defined(PADDLE_WITH_HIP)
  DenseTensor scale_trans(scale.type());
  if (group_size > 0) {
    scale_trans.Resize({scale.dims()[1], scale.dims()[0]});
    dev_ctx.template Alloc<T>(&scale_trans);
    std::vector<int> axis = {1, 0};
    funcs::Transpose<Context, T, 2> trans;
    trans(dev_ctx, scale, &scale_trans, axis);
  }
  auto out_dims = out->dims();
  dev_ctx.template Alloc<T>(out);
  WeightDequantize<T, Context>(
      dev_ctx, x, group_size > 0 ? scale_trans : scale, algo, group_size, out);
  out->Resize({{out_dims[1], out_dims[0]}});
  auto out_tmp = Transpose<T, Context>(dev_ctx, *out, {1, 0});
  out->ShareDataWith(out_tmp);
#else
  PADDLE_THROW(
      common::errors::PreconditionNotMet("Not compiled with WITH_CUTLASS=ON"));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(weight_dequantize,
                   GPU,
                   ALL_LAYOUT,
                   phi::WeightDequantizeKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
