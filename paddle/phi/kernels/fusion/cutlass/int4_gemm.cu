// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "cutlass/numeric_conversion.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/fusion/cutlass/int4_gemm/int4_gemm_decl.h"
#include "paddle/phi/kernels/fusion/cutlass/int4_gemm/int4_gemm_util.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {
namespace fusion {
namespace cutlass_gemm_internal {
template <typename T, typename Context>
void Int4GemmKernel(const Context &ctx,
                    const DenseTensor &x,
                    const DenseTensor &y,
                    const DenseTensor &bias,
                    const bool trans_x,
                    const bool trans_y,
                    const std::string &activation,
                    DenseTensor *out) {
  ctx.template Alloc<T>(out);
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto bias_dims = bias.dims();
  auto out_dims = out->dims();
  CHECK_EQ(x_dims.size() == 2UL, true);
  CHECK_EQ(y_dims.size() == 2UL, true);
  CHECK_EQ(bias_dims.size() == 1UL, true);

  CHECK_EQ(out_dims.size() == 2UL, true);

  const int m = trans_x ? x_dims[1] : x_dims[0];
  const int kx = trans_x ? x_dims[0] : x_dims[1];
  const int ky = trans_y ? y_dims[1] : y_dims[0];
  const int n = trans_y ? y_dims[0] : y_dims[1];
  const int mk = m * kx;
  const int kn = ky * n;
  const int mn = m * n;

  CHECK_EQ(kx, ky);
  CHECK_EQ(m % 8, 0);
  CHECK_EQ(kx % 8, 0);
  CHECK_EQ(n % 8, 0);

  int sm = getSMVersion();
  if (sm != 75 && sm != 80) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Cutlass does not support int4 gemm on sm %d", sm));
  }

  auto stream = ctx.stream();

  size_t x_bytes = m * kx / 2;
  size_t y_bytes = ky * n / 2;
  size_t out_bytes = m * n * 4;

  auto tmp_x = paddle::memory::Alloc(
      ctx.GetPlace(),
      x_bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  auto tmp_y = paddle::memory::Alloc(
      ctx.GetPlace(),
      y_bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  auto tmp_out = paddle::memory::Alloc(
      ctx.GetPlace(),
      out_bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  auto tmp_bias = paddle::memory::Alloc(
      ctx.GetPlace(),
      out_bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(stream)));

  ConvertDataToInt4<T, Context>(
      ctx, x, reinterpret_cast<cutlass::int4b_t *>(tmp_x->ptr()), mk, trans_x);
  // Int4 GEMM need y in column-major,so action on y is opposite
  ConvertDataToInt4<T, Context>(
      ctx, y, reinterpret_cast<cutlass::int4b_t *>(tmp_y->ptr()), kn, !trans_y);

  dim3 gridb(128);
  dim3 blockb(n);
  ExpendKernel<int32_t>
      <<<gridb, blockb>>>(reinterpret_cast<const int32_t *>(bias.data()),
                          reinterpret_cast<int32_t *>(tmp_bias->ptr()),
                          n,
                          m,
                          0);

  GemmAllParams params = {
      reinterpret_cast<const cutlass::int4b_t *>(tmp_x->ptr()),
      reinterpret_cast<const cutlass::int4b_t *>(tmp_y->ptr()),
      reinterpret_cast<const int32_t *>(tmp_bias->ptr()),
      reinterpret_cast<int32_t *>(tmp_out->ptr()),
      1,
      m,
      n,
      kx,
      &ctx};
  if (activation == "none") {
    Int4GemmBias(params, sm);
  } else if (activation == "relu") {
    Int4GemmRelu(params, sm);
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Cutlass dose not support this activation on int4: %s.",
        activation.c_str()));
  }

  constexpr int blocko_ = 256;
  dim3 grido((mn + blocko_ - 1) / blocko_);
  dim3 blocko(blocko_);

  DynamicConvert<T, int32_t>
      <<<grido, blocko>>>(reinterpret_cast<const int32_t *>(tmp_out->ptr()),
                          reinterpret_cast<T *>(out->data()),
                          mn);
}
}  // namespace cutlass_gemm_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(int4_gemm_cutlass,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_gemm_internal::Int4GemmKernel,
                   int32_t) {}
