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

#include "paddle/phi/kernels/quant_for_compress_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/impl/quant_for_compress_kernel_impl.h"

namespace phi {

template <typename DeviceContext, typename T, typename D, int bits>
void quant_compute(const DeviceContext& dev_ctx,
                   const DenseTensor& x,
                   DenseTensor* out,
                   DenseTensor* scale,
                   const std::string& layout) {
  const auto x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "the x tensor of quant op must be 2D, but got[%d]", x_dims.size()));
  size_t m = x_dims[0];
  size_t n = x_dims[1];
  int64_t num = x.numel();
  DDim dims = {num};
  const T* x_data = x.data<T>();
  D* out_data = out->data<D>();
  float* scale_data = scale->data<float>();

  DenseTensor x_int(out->type());
  x_int.Resize({static_cast<int64_t>(m), static_cast<int64_t>(n)});
  dev_ctx.template Alloc<D>(&x_int);
  D* x_int_data = x_int.data<D>();

  DenseTensor int_processed(out->type());
  int_processed.Resize(dims);
  dev_ctx.template Alloc<D>(&int_processed);

  D* int_processed_data = int_processed.data<D>();
  DenseTensor int_processed_2(out->type());
  int_processed_2.Resize(out->dims());
  dev_ctx.template Alloc<D>(&int_processed_2);
  D* int_processed_2_data = int_processed_2.data<D>();

  per_channel_scale(scale_data, x_data, m, n);

  per_channel_quant<T, bits>(x_int_data, x_data, scale_data, m, n);
  if (layout == "weight_only") {
    permute_B_rows_for_mixed_gemm<bits>(
        int_processed_data, x_int_data, std::vector<size_t>{m, n}, (int64_t)80);
    subbyte_transpose_impl<bits>(
        int_processed_2_data, int_processed_data, std::vector<size_t>{m, n});
    interleave_column_major_tensor<bits>(
        out_data, int_processed_2_data, std::vector<size_t>{m, n});
    add_bias_and_interleave_inplace<bits>(out_data, num);
  } else if (layout == "llm.int8") {
    std::vector<int> axis = {1, 0};
    funcs::Transpose<DeviceContext, int8_t, 2> trans;
    trans(dev_ctx, x_int, out, axis);
  } else {
    phi::errors::InvalidArgument(
        "The layout must be weight_only or llm.int8, but got %s", layout);
  }
}

template <typename T, typename Context>
void QuantForCompressKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            int bits,
                            const std::string& layout,
                            DenseTensor* out,
                            DenseTensor* scale) {
  if (bits == 8) {
    dev_ctx.template Alloc<int8_t>(out);
    dev_ctx.template Alloc<float>(scale);
    quant_compute<Context, T, int8_t, 8>(dev_ctx, x, out, scale, layout);
  } else if (bits == 4 && layout == "weight_only") {
    dev_ctx.template Alloc<int8_t>(out);
    dev_ctx.template Alloc<float>(scale);
    quant_compute<Context, T, int8_t, 4>(dev_ctx, x, out, scale, layout);
  } else {
    phi::errors::Unimplemented(
        "The bits only support 8 or weight_only 4, but got[%s] [%d]",
        layout,
        bits);
  }
  // VLOG(0) << "x: " << x.dtype() << x;
  // VLOG(0) << "out: " << out->dtype() << *out;
}

}  // namespace phi

PD_REGISTER_KERNEL(quant_for_compress,
                   CPU,
                   ALL_LAYOUT,
                   phi::QuantForCompressKernel,
                   phi::dtype::float16) {}
