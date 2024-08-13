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

#include "paddle/phi/kernels/weight_quantize_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/impl/weight_quantize_kernel_impl.h"

namespace phi {

template <typename DeviceContext,
          typename T,
          typename D,
          int bits,
          typename ScaleT = T>
void quant_compute(const DeviceContext& dev_ctx,
                   const DenseTensor& x,
                   DenseTensor* out,
                   DenseTensor* scale,
                   const std::string& algo,
                   const int32_t arch,
                   const int32_t group_size) {
#ifndef PADDLE_WITH_HIP
  PADDLE_ENFORCE_EQ(
      ((arch == 70) || (arch == 75) || (arch == 80) || (arch == 86) ||
       (arch == 89) || (arch == 90)),
      true,
      common::errors::InvalidArgument(
          "Currently, arch only support 70, 75, 80, 86, 89, 90."));

#endif
  const auto x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      2,
      common::errors::InvalidArgument(
          "the x tensor of quant op must be 2D, but got[%d]", x_dims.size()));
  size_t m = x_dims[0];
  size_t n = x_dims[1];
  int64_t num = x.numel();
  DDim dims = {num};
  const T* x_data = x.data<T>();
  D* out_data = out->data<D>();
  ScaleT* scale_data = scale->data<ScaleT>();

  DenseTensor x_int(out->type());

#ifdef PADDLE_WITH_HIP
  x_int.Resize({static_cast<int64_t>(m), static_cast<int64_t>(n)});
#else
  if ((arch == 80) || (arch == 75) || (arch == 86) || (arch == 89) ||
      (arch == 90)) {
    x_int.Resize({static_cast<int64_t>(m), static_cast<int64_t>(n)});
  } else {
    // phi::Copy may change tensor meta info, here we transpose the quanted
    // data's shape.
    x_int.Resize({static_cast<int64_t>(n), static_cast<int64_t>(m)});
  }
#endif

  dev_ctx.template Alloc<D>(&x_int);
  D* x_int_data = x_int.data<D>();

#ifdef PADDLE_WITH_HIP
  DenseTensor x_int_tmp(x_int.type());
  x_int_tmp.Resize({static_cast<int64_t>(m), static_cast<int64_t>(n / 2)});
  dev_ctx.template Alloc<D>(&x_int_tmp);
  D* x_int_tmp_data = x_int_tmp.data<D>();
#else
  DenseTensor int_processed(out->type());
  int_processed.Resize(dims);
  dev_ctx.template Alloc<D>(&int_processed);

  D* int_processed_data = int_processed.data<D>();
  DenseTensor int_processed_2(out->type());
  int_processed_2.Resize(out->dims());
  dev_ctx.template Alloc<D>(&int_processed_2);
  D* int_processed_2_data = int_processed_2.data<D>();
#endif

  if (group_size == -1) {
    per_channel_scale(scale_data, x_data, m, n, bits == 8 ? 127.0f : 7.0f);
    per_channel_quant<T, bits>(x_int_data, x_data, scale_data, m, n);
  } else {
    group_wise_scale(scale_data,
                     x_data,
                     m,
                     n,
                     bits == 8 ? 127.0f : 7.0f,
                     static_cast<size_t>(group_size));

    group_wise_quant<T, bits>(x_int_data, x_data, scale_data, m, n, group_size);
  }
  if (algo == "llm.int8") {
    std::vector<int> axis = {1, 0};
    funcs::Transpose<DeviceContext, int8_t, 2> trans;
    trans(dev_ctx, x_int, out, axis);
  } else {
#ifdef PADDLE_WITH_HIP
    if (bits == 8) {
      std::vector<int> axis = {1, 0};
      funcs::Transpose<DeviceContext, int8_t, 2> trans;
      trans(dev_ctx, x_int, out, axis);
    } else {
      for (int i = 0; i < out->numel(); ++i) {
        x_int_tmp_data[i] = x_int_data[i];
      }
      std::vector<int> axis = {1, 0};
      funcs::Transpose<DeviceContext, int8_t, 2> trans;
      trans(dev_ctx, x_int_tmp, out, axis);
    }
#else
    if (arch == 70) {
      // Note(Zhengzekang): In sm70, we only need RowMajor layout, just add bias
      // to make it unsigned.
      add_bias_and_interleave_inplace<bits>(x_int_data, num);
      // phi::Copy break the shape of int4 output, use naive copy;
      // only left half of x_int data is valid in int4 mode
      for (int i = 0; i < out->numel(); ++i) {
        out_data[i] = x_int_data[i];
      }
    } else if ((arch == 90) || (arch == 89) || (arch == 86) || (arch == 80) ||
               (arch == 75)) {
      permute_B_rows_for_mixed_gemm<bits>(
          int_processed_data, x_int_data, std::vector<size_t>{m, n});
      subbyte_transpose_impl<bits>(
          int_processed_2_data, int_processed_data, std::vector<size_t>{m, n});
      interleave_column_major_tensor<bits>(
          out_data, int_processed_2_data, std::vector<size_t>{m, n});
      add_bias_and_interleave_inplace<bits>(out_data, num);
    }
#endif
  }
}

template <typename T, typename Context>
void WeightQuantizeKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const std::string& algo,
                          const int32_t arch,
                          const int32_t group_size,
                          DenseTensor* out,
                          DenseTensor* scale) {
  dev_ctx.template Alloc<int8_t>(out);
  if (algo == "weight_only_int8") {
    dev_ctx.template Alloc<T>(scale);
    quant_compute<Context, T, int8_t, 8>(
        dev_ctx, x, out, scale, algo, arch, group_size);
  } else if (algo == "llm.int8") {
    dev_ctx.template Alloc<float>(scale);
    quant_compute<Context, T, int8_t, 8, float>(
        dev_ctx, x, out, scale, algo, arch, group_size);
  } else if (algo == "weight_only_int4") {
    dev_ctx.template Alloc<T>(scale);
    quant_compute<Context, T, int8_t, 4>(
        dev_ctx, x, out, scale, algo, arch, group_size);
  } else {
    common::errors::Unimplemented(
        "The algo must be in ['weight_only_int8', 'weight_only_int4', "
        "'llm.int8'], but got[%s]",
        algo);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(weight_quantize,
                   CPU,
                   ALL_LAYOUT,
                   phi::WeightQuantizeKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
