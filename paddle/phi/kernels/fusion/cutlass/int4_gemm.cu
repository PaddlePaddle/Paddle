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

#include "cutlass/numeric_conversion.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/fusion/cutlass/int4_gemm/int4_gemm_decl.h"
#include "paddle/phi/kernels/fusion/cutlass/int4_gemm/int4_gemm_util.h"

namespace phi {
namespace fusion {
namespace cutlass_gemm_internal {
template <typename T, typename Context>
void Int4GemmKernel(const Context &ctx,
                    const DenseTensor &x,
                    const DenseTensor &y,
                    const DenseTensor &bias,
                    DenseTensor *out,
                    const bool &trans_x,
                    const bool &trans_y,
                    const std::string &activation) {
  ctx.template Alloc<T>(output);
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto bias_dims = bias.dims();
  auto out_dims = out->dims();
  CHECK_EQ(x_dims.size() == 2UL, true);
  CHECK_EQ(y_dims.size() == 2UL, true);
  CHECK_EQ(bias_dims.size() == 1UL, true);

  CHECK_EQ(out_dims.size() == 2UL, true);

  const int m = x_dims[0];
  const int kx = x_dims[1];
  const int ky = y_dims[0];
  const int n = y_dims[1];

  CHECK_EQ(kx, ky);

  int sm = getSMVersion();
  if (sm != 75 && sm != 80) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Cutlass does not support int4 gemm on sm %d", sm));
  }
  auto stream = ctx.stream();

  size_t x_bytes = m * kx / 2;
  size_t y_bytes = ky * n / 2;
  size_t out_bytes = m * n * 4;

  paddle::memory::allocation::AllocationPtr tmp_x = paddle::memory::Alloc(
      ctx.GetPlace(),
      x_bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  paddle::memory::allocation::AllocationPtr tmp_y = paddle::memory::Alloc(
      ctx.GetPlace(),
      y_bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  paddle::memory::allocation::AllocationPtr tmp_out = paddle::memory::Alloc(
      ctx.GetPlace(),
      out_bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(stream)));

  cutlass::Array<T, m *kx> *source_x =
      reinterpret_cast<cutlass::Array<T, m * kx> *>(x.data());
  cutlass::Array<T, kx *n> *source_y =
      reinterpret_cast<cutlass::Array<T, ky * n> *>(y.data());

  cutlass::NumericArrayConverter<cutlass::int4b_t, T, m * kx> convert_x;
  cutlass::NumericArrayConverter<cutlass::int4b_t, T, ky * n> convert_y;

  cutlass::Array<cutlass::int4b_t, m *kx> *destination_x = tmp_x->ptr();
  *destination_x = convert_x(*source_x);
  cutlass::Array<cutlass::int4b_t, ky *n> *destination_y = tmp_y->ptr();
  *destination_x = convert_y(*source_y);
  int32_t *destination_output = tmp_out->ptr();

  GemmAllParams params = {
      reinterpret_cast<const cutlass::int4b_t *>(destination_x->data()),
      reinterpret_cast<const cutlass::int4b_t *>(destination_y->data()),
      reinterpret_cast<const int32_t *>(bias.data()),
      destination_output,
      1,
      m,
      n,
      k,
      &ctx};
  if (activation == "identity") {
    Int4GemmBias(params, sm);
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Cutlass dose not support this activation on int4: %s.",
        activation.c_str()));
  }
  cutlass::Array<int32_t, m *n> *source_out = destination_output;
  cutlass::NumericArrayConverter<T, int32_t, m * n> convert_out;
  *out = convert_out(*source_out);
  out->set_layout(ALL_LAYOUT);
}
}  // namespace cutlass_gemm_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(int4_gemm_cutlass,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_gemm_internal::Int4GemmKernel,
                   int8_t) {}
