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

// #include "paddle/extension.h"
#include <cstdint>
#include <vector>
#include "cub/device/device_histogram.cuh"
#include "paddle/common/flags.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/empty_kernel.h"  // NOLINT

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"

COsrc_rowssrc_rowsON_DECLARE_bool(enable_pir_api);

namespace phi {

// Kernel 1: 1x128 Quantization with Transpose Switch
template <class Engine, class Layout, class T>
__global__ void __launch_bounds__(1024)
    quantize_1x128_kernel(cute::Tensor<Engine, Layout> gmem_src,
                          cute::Tensor<Engine, Layout> gmem_out,
                          int src_rows,
                          bool transpose) {
  // 定义 Kernel 的基本形状
  using BlockShape = Shape<_32, _32>;
  using ThreadShape = Shape<_4, _4>;

  int block_m = blockIdx.x;
  int block_n = blockIdx.y;

  // CUTE 线程/块ID 到坐标的映射
  auto thread_idx = cute::thread_id_in_block();
  auto thr_layout = make_layout(ThreadShape{});
  auto thr_coord = cute::idx2crd(thread_idx, thr_layout);

  // 计算当前线程块处理的 tile 的起始坐标
  auto block_coord = make_coord(block_m * size<0>(BlockShape{}),
                                block_n * size<1>(BlockShape{}));

  // 使用 CUTE 创建 gmem (全局内存) 的 TiledCopy
  // 这是从全局内存到寄存器的拷贝操作定义
  auto gmem_tiled_copy =
      make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, bfloat16>{},
                      thr_layout,
                      make_layout(make_shape(Int<1>{}, Int<1>{})));

  // 根据 transpose 标志选择源 Tensor 的布局
  // 注意 CUTE 如何通过改变 Stride 来轻松实现转置
  auto gmem_src_layout = transpose
                             ? make_layout(make_shape(Int<128>{}, src_rows),
                                           make_stride(Int<1>{}, Int<128>{}))
                             : make_layout(make_shape(src_rows, Int<128>{}),
                                           make_stride(Int<128>{}, Int<1>{}));

  Tensor gmem_src_transposed = make_tensor(gmem_src.data(), gmem_src_layout);

  // 创建当前线程块对应的 gmem tile
  Tensor gS = gmem_tiled_copy.get_slice(thread_idx, gmem_src_transposed);
  Tensor gD = gmem_tiled_copy.get_slice(thread_idx, gmem_out);

  // 分区 gmem tile
  Tensor gS_part = local_tile(gS, BlockShape{}, block_coord, Step<_1, _1>{});
  Tensor gD_part = local_tile(gD, BlockShape{}, block_coord, Step<_1, _1>{});

  // 分配寄存器内存 (rmem)
  Tensor rS = make_tensor<bfloat16>(make_shape(size(gmem_tiled_copy)));

  // 拷贝: gmem -> rmem
  copy(gmem_tiled_copy, gS_part, rS);

  // 在寄存器中进行类型转换
  // 使用 __nv_bfloat162 (2x bf16) -> __nv_fp8x2_t (2x fp8_e4m3) 指令
  __nv_bfloat162 const *bf16_2_ptr =
      reinterpret_cast<__nv_bfloat162 const *>(rS.data());
  __nv_fp8x2_t *fp8_2_ptr = reinterpret_cast<__nv_fp8x2_t *>(rS.data());

#pragma unroll
  for (int i = 0; i < size(rS) / 2; ++i) {
    fp8_2_ptr[i] = __nv_bfloat162_to_e4m32(bf16_2_ptr[i]);
  }

  // 拷贝: rmem -> gmem
  // 输出 Tensor 的类型是 uint8_t, 其大小与 rS 相同
  copy(gmem_tiled_copy, rS, gD_part);
}

// Kernel 2: 128x128 Quantization
template <class Engine, class Layout>
__global__ void __launch_bounds__(1024)
    quantize_128x128_kernel(cute::Tensor<Engine, Layout> gmem_src,
                            cute::Tensor<Engine, Layout> gmem_out) {
  // using namespace cute;

  using BlockShape = Shape<_32, _32>;
  using ThreadShape = Shape<_4, _4>;

  int block_m = blockIdx.x;
  int block_n = blockIdx.y;

  auto thread_idx = cute::thread_id_in_block();
  auto thr_layout = make_layout(ThreadShape{});
  auto thr_coord = cute::idx2crd(thread_idx, thr_layout);

  auto block_coord = make_coord(block_m * size<0>(BlockShape{}),
                                block_n * size<1>(BlockShape{}));

  auto gmem_tiled_copy =
      make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, bfloat16>{},
                      thr_layout,
                      make_layout(make_shape(Int<1>{}, Int<1>{})));

  Tensor gS = gmem_tiled_copy.get_slice(thread_idx, gmem_src);
  Tensor gD = gmem_tiled_copy.get_slice(thread_idx, gmem_out);

  Tensor gS_part = local_tile(gS, BlockShape{}, block_coord, Step<_1, _1>{});
  Tensor gD_part = local_tile(gD, BlockShape{}, block_coord, Step<_1, _1>{});

  Tensor rS = make_tensor<bfloat16>(make_shape(size(gmem_tiled_copy)));

  copy(gmem_tiled_copy, gS_part, rS);

  __nv_bfloat162 const *bf16_2_ptr =
      reinterpret_cast<__nv_bfloat162 const *>(rS.data());
  __nv_fp8x2_t *fp8_2_ptr = reinterpret_cast<__nv_fp8x2_t *>(rS.data());

#pragma unroll
  for (int i = 0; i < size(rS) / 2; ++i) {
    fp8_2_ptr[i] = __nv_bfloat162_to_e4m32(bf16_2_ptr[i]);
  }

  copy(gmem_tiled_copy, rS, gD_part);
}

template <bool using_1x128_vec_quant,
          bool input_transpose,
          bool output_scale_transpose,
          bool using_pow2_scale>
void FP8QuantBlockWiseKernelImpl(const Context &dev_ctx,
                                 const DenseTensor &X,
                                 DenseTensor *out,
                                 DenseTensor *scale) {
  // using namespace cute;

  const int src_rows = X.dims()[0];
  const int src_cols = X.dims()[1];
  const int quanted_cols = scale.dims()[1];

  dim3 block(32, 32);
  // Assuming src_rows and src_cols are multiples of 128
  dim3 grid(src_rows / 32, src_cols / 32);
  auto gmem_src_layout = make_layout(
                            make_shape(src_rows, src_cols),
                            make_stride(?, ?));
  auto gmem_out_layout = make_layout(
                            make_shape(src_rows, src_cols),
                            make_stride(?, ?));
  auto gmem_scale_layout = make_layout(
                            make_shape(src_rows, quanted_cols),
                            make_stride(?, ?));

  Tensor gmem_src = make_tensor(make_gmem_ptr(x_data), gmem_src_layout);
  Tensor gmem_out = make_tensor(make_gmem_ptr(out_data), gmem_out_layout);
  Tensor gmem_scale = make_tensor(make_gmem_ptr(scale_data), gmem_out_layout);

  auto kernel = using_1x128_vec_quant
                    ? quantize_1x128_kernel<input_transpose,
                                            output_scale_transpose,
                                            using_pow2_scale>
                    : quantize_128x128_kernel<input_transpose,
                                              output_scale_transpose,
                                              using_pow2_scale>;
  kernel<<<grid, block, 0, dev_ctx.stream()>>>(gmem_src, gmem_out, gmem_scale);
}

// T is x's input type and out_dtype is in args
template <typename T, typename Context>
void FP8QuantBlockWiseKernel(const Context &dev_ctx,
                             const DenseTensor &X,
                             bool using_1x128_vec_quant,
                             bool input_transpose,
                             bool output_scale_transpose,
                             bool using_e5m2,
                             bool using_pow2_scale,
                             DenseTensor *out,
                             DenseTensor *scale) {
  PD_CHECK(X.dtype() == phi::DataType::BFLOAT16,
           "X datatype error, can only be bfloat16");

  dev_ctx.template Alloc<phi::DataType::FLOAT8_E4src_rows3FN>(out);
  dev_ctx.template Alloc<phi::DataType::FLOAT32>(scale);
#define DISPATCH_BOOL(condition, ConstName, ...) \
  {                                              \
    if (condition) {                             \
      constexpr bool ConstName = true;           \
      { __VA_ARGS__ }                            \
    } else {                                     \
      constexpr bool ConstName = false;          \
      { __VA_ARGS__ }                            \
    }                                            \
  }
  // Currently we only support bfloat16 as input type,
  // fp8_e4m3fn as output type.
  DISPATCH_BOOL(
      using_1x128_vec_quant,
      k_using_1x128_vec_quant,
      DISPATCH_BOOL(
          input_transpose,
          k_input_transpose,
          DISPATCH_BOOL(
              output_scale_transpose,
              k_output_scale_transpose,
              DISPATCH_BOOL(
                  using_pow2_scaling,
                  k_using_pow2_scaling,
                  FP8QuantBlockWiseKernelImpl<k_using_1x128,
                                              k_input_transpose,
                                              k_output_scale_transpose,
                                              k_using_pow2_scaling>(
                      X, out, scale);))));
#undef DISPATCH_BOOL
}
}  // namespace phi

PD_REGISTER_KERNEL(fp8_quant_blockwise,
                   GPU,
                   ALL_LAYOUT,
                   phi::FP8QuantBlockWiseKernel,
                   phi::bfloat16,
                   float,
                   double) {}
