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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include <limits>
#include <set>
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)
#include "paddle/phi/kernels/funcs/dims_simplifier.h"

#endif

COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_stride_compute_kernel);

namespace phi {

template <typename Context>
phi::DenseTensor Tensor2Contiguous(const Context &dev_ctx,
                                   const phi::DenseTensor &tensor) {
  phi::DenseTensor dense_out;
  phi::MetaTensor meta_input(tensor);
  phi::MetaTensor meta_out(&dense_out);
  UnchangedInferMeta(meta_input, &meta_out);
  PD_VISIT_ALL_TYPES(tensor.dtype(), "Tensor2Contiguous", ([&] {
                       phi::ContiguousKernel<data_t, Context>(
                           dev_ctx, tensor, &dense_out);
                     }));
  return dense_out;
}

/**
 * Check if tensor is only transposed and return the original
 * contiguous shape/stride and transpose axis mapping.
 */
inline bool is_only_transposed_tensor(const DDim &shape,
                                      const DDim &stride,
                                      const uint64_t &offset,
                                      DDim *src_shape,
                                      DDim *src_stride,
                                      std::vector<int> *axis) {
  if (offset != 0) {
    return false;
  }
  std::set<int> visited_idx;
  axis->resize(stride.size());
  for (int i = 0; i < stride.size(); i++) {
    int64_t max_num = 0;
    int max_idx = -1;
    for (int j = 0; j < stride.size(); j++) {
      if (visited_idx.count(j)) {
        continue;
      }
      if (stride[j] < 1) {
        return false;
      }
      if (stride[j] > max_num) {
        max_num = stride[j];
        max_idx = j;
      }
    }
    if (max_idx == -1) {
      return false;
    }
    if (i != 0 && (*src_stride)[i - 1] == max_num) {
      return false;
    }
    visited_idx.insert(max_idx);
    (*src_stride)[i] = max_num;
    (*src_shape)[i] = shape[max_idx];
    (*axis)[max_idx] = i;
  }

  if (DenseTensorMeta::calc_strides(*src_shape) == *src_stride) {
    return true;
  } else {
    return false;
  }
}

template <typename T, typename Context>
void MatmulStrideKernel(const Context &dev_ctx,
                        const DenseTensor &x,
                        const DenseTensor &y,
                        bool transpose_x,
                        bool transpose_y,
                        DenseTensor *out) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  DenseTensor x_;
  DenseTensor y_;

  // if (out && (out->IsSharedBufferWith(x) || out->IsSharedBufferWith(y))) {
  //   printf("inplace found\n");
  // } else {
  //   printf("inplace not found\n");
  // }

  // if (!FLAGS_use_stride_compute_kernel) {
  if (!x.meta().is_contiguous()) {
    x_ = Tensor2Contiguous<Context>(dev_ctx, x);
  } else {
    x_ = x;
  }
  if (!y.meta().is_contiguous()) {
    y_ = Tensor2Contiguous<Context>(dev_ctx, y);
  } else {
    y_ = y;
  }
  // } else {
  //   x_ = x;
  //   y_ = y;
  // }

  if (x_.meta().is_contiguous() && y_.meta().is_contiguous()) {
    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
    phi::MatmulKernel<T, Context>(
        dev_ctx, x_, y_, transpose_x, transpose_y, out);
    return;
  }

  // if (!FLAGS_use_stride_compute_kernel) {
  //   PADDLE_THROW(
  //       common::errors::Fatal("FLAGS_use_stride_compute_kernel is closed. "
  //                             "Kernel using DenseTensorIterator "
  //                             "be called, something wrong has happened!"));
  // }

  // auto x_meta = x_.meta();
  // DDim x_stride = x_meta.strides;
  // DDim x_shape = x_meta.dims;
  // std::vector<int> x_axis;
  // auto y_meta = y_.meta();
  // DDim y_stride = y_meta.strides;
  // DDim y_shape = y_meta.dims;
  // std::vector<int> y_axis;

  // printf("x dims\n");
  // for (int i=0; i<x.dims().size(); i++) {
  //   printf("%d ", x.dims()[i]);
  // }
  // printf("\n");

  // printf("x strides\n");
  // for (int i=0; i<x.strides().size(); i++) {
  //   printf("%d ", x.strides()[i]);
  // }
  // printf("\n");

  // printf("x offset:%d\n", x.offset());

  // printf("y dims\n");
  // for (int i=0; i<y.dims().size(); i++) {
  //   printf("%d ", y.dims()[i]);
  // }
  // printf("\n");

  // printf("y strides\n");
  // for (int i=0; i<y.strides().size(); i++) {
  //   printf("%d ", y.strides()[i]);
  // }
  // printf("\n");

  // printf("y offset:%d\n", y.offset());

  // printf("out offset:%d\n", out->offset());

  // DenseTensor x_trans_res;

  // if (!x.meta().is_contiguous() && is_only_transposed_tensor(x_meta.dims,
  //                                                            x_meta.strides,
  //                                                            x_meta.offset,
  //                                                            &x_shape,
  //                                                            &x_stride,
  //                                                            &x_axis)) {
  //   printf("x trans axis\n");
  //   for (int i=0; i<x_axis.size(); i++) {
  //     printf("%d ", x_axis[i]);
  //   }
  //   printf("\n");
  //   auto x_trans_dims = x_axis.size();
  //   if (x_axis[x_trans_dims - 1] == x_trans_dims - 2 &&
  //       x_axis[x_trans_dims - 2] == x_trans_dims - 1) {
  //     printf("trans x\n");
  //     transpose_x = !transpose_x;
  //   DenseTensor tmp_tensor = x_;
  //   phi::DenseTensorMeta tmp_meta = x_.meta();
  //   tmp_meta.strides = x_stride;
  //   tmp_meta.dims = x_shape;
  //   tmp_tensor.set_meta(tmp_meta);
  //   TransposeStridedKernel<Context>(dev_ctx, tmp_tensor, x_axis,
  //   &x_trans_res); } else {
  //     x_trans_res = x_;
  //   }

  // printf("x_ dims\n");
  // for (int i=0; i<x_.dims().size(); i++) {
  //   printf("%d ", x_.dims()[i]);
  // }
  // printf("\n");

  // printf("x_ strides\n");
  // for (int i=0; i<x_.strides().size(); i++) {
  //   printf("%d ", x_.strides()[i]);
  // }
  // printf("\n");

  // } else {
  //   x_trans_res = x_;
  // }

  // if (!x_trans_res.meta().is_contiguous()) {
  //   printf("x trans to conti in here\n");
  //   x_trans_res = Tensor2Contiguous<Context>(dev_ctx, x);
  // }

  // printf("x_ final dims\n");
  // for (int i=0; i<x_.dims().size(); i++) {
  //   printf("%d ", x_.dims()[i]);
  // }
  // printf("\n");

  // printf("x_ final strides\n");
  // for (int i=0; i<x_.strides().size(); i++) {
  //   printf("%d ", x_.strides()[i]);
  // }
  // printf("\n");

  // DenseTensor y_trans_res;

  // if (!y.meta().is_contiguous() && is_only_transposed_tensor(y_meta.dims,
  //                                                            y_meta.strides,
  //                                                            y_meta.offset,
  //                                                            &y_shape,
  //                                                            &y_stride,
  //                                                            &y_axis)) {
  //   // printf("y trans axis\n");
  //   // for (int i=0; i<y_axis.size(); i++) {
  //   //   printf("%d ", y_axis[i]);
  //   // }
  //   // printf("\n");
  //   auto y_trans_dims = y_axis.size();
  //   if (y_axis[y_trans_dims - 1] == y_trans_dims - 2 &&
  //       y_axis[y_trans_dims - 2] == y_trans_dims - 1) {
  //     printf("trans y\n");
  //     // transpose_y = !transpose_y;
  //     // y_meta.dims = y_shape;
  //     // y_meta.strides = y_stride;
  //     // y_.set_meta(y_meta);

  //   printf("trans y shape\n");
  // for (int i=0; i<y_shape.size(); i++) {
  //   printf("%d ", y_shape[i]);
  // }
  // printf("\n");

  //   printf("trans y_stride\n");
  // for (int i=0; i<y_stride.size(); i++) {
  //   printf("%d ", y_stride[i]);
  // }
  // printf("\n");

  //   // transpose_y = !transpose_y;
  //   // DenseTensor tmp_tensor = y_;
  //   // phi::DenseTensorMeta tmp_meta = y_.meta();
  //   // tmp_meta.strides = y_stride;
  //   // tmp_meta.dims = y_shape;
  //   // y_trans_res.set_meta(tmp_meta);
  //   // TransposeStridedKernel<Context>(dev_ctx, y_, y_axis, &y_trans_res);
  //   y_trans_res = Tensor2Contiguous<Context>(dev_ctx, y_);

  // printf("y_ after trans dims\n");
  // for (int i=0; i<y_trans_res.dims().size(); i++) {
  //   printf("%d ", y_trans_res.dims()[i]);
  // }
  // printf("\n");

  // printf("y_ after trans strides\n");
  // for (int i=0; i<y_trans_res.strides().size(); i++) {
  //   printf("%d ", y_trans_res.strides()[i]);
  // }
  // printf("\n");

  //   } else {
  //     y_trans_res = y_;
  //   }
  // printf("y_ dims\n");
  // for (int i=0; i<y_.dims().size(); i++) {
  //   printf("%d ", y_.dims()[i]);
  // }
  // printf("\n");

  // printf("y_ strides\n");
  // for (int i=0; i<y_.strides().size(); i++) {
  //   printf("%d ", y_.strides()[i]);
  // }
  // printf("\n");
  // } else {
  //   y_trans_res = y_;
  // }

  // if (!y_trans_res.meta().is_contiguous()) {
  //   printf("y trans to conti in here\n");
  //   y_trans_res = Tensor2Contiguous<Context>(dev_ctx, y);
  // }

  // printf("y_ final dims\n");
  // for (int i=0; i<y_trans_res.dims().size(); i++) {
  //   printf("%d ", y_trans_res.dims()[i]);
  // }
  // printf("\n");

  // printf("y_ final strides\n");
  // for (int i=0; i<y_trans_res.strides().size(); i++) {
  //   printf("%d ", y_trans_res.strides()[i]);
  // }
  // printf("\n");

  // auto meta = out->meta();
  // meta.strides = meta.calc_strides(out->dims());
  // out->set_meta(meta);
  // phi::MatmulKernel<T, Context>(dev_ctx, x_trans_res, y_trans_res,
  // transpose_x, transpose_y, out);
}

}  // namespace phi

#if CUDA_VERSION >= 12010 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
PD_REGISTER_KERNEL(matmul,
                   GPU,
                   STRIDED,
                   phi::MatmulStrideKernel,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   phi::float8_e4m3fn,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128,
                   int8_t) {
#else
PD_REGISTER_KERNEL(matmul,
                   GPU,
                   STRIDED,
                   phi::MatmulStrideKernel,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128,
                   int8_t) {
#endif
  if (kernel_key.dtype() == phi::DataType::INT8) {
    kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
  }
  if (kernel_key.dtype() == phi::DataType::FLOAT8_E4M3FN) {
    kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT16);
  }
}

#endif
