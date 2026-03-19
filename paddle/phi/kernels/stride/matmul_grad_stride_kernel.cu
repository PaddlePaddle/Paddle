// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <set>

#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/matmul_grad_kernel.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

inline void PrepareStridedOut(DenseTensor* out) {
  if (out == nullptr) {
    return;
  }
  auto meta = out->meta();
  meta.strides = meta.calc_strides(out->dims());
  out->set_meta(meta);
}

template <typename Context>
DenseTensor Tensor2Contiguous(const Context& dev_ctx,
                              const DenseTensor& tensor) {
  DenseTensor dense_out;
  MetaTensor meta_input(tensor);
  MetaTensor meta_out(&dense_out);
  UnchangedInferMeta(meta_input, &meta_out);
  PD_VISIT_ALL_TYPES(tensor.dtype(), "Tensor2Contiguous", ([&] {
                       phi::ContiguousKernel<data_t, Context>(
                           dev_ctx, tensor, &dense_out);
                     }));
  return dense_out;
}

inline bool IsOnlyTransposedTensor(const DenseTensor& tensor,
                                   DDim* src_shape,
                                   DDim* src_stride,
                                   std::vector<int>* axis) {
  const auto& meta = tensor.meta();
  if (meta.dims.size() < 2 || meta.offset != 0 ||
      meta.strides == DenseTensorMeta::calc_strides(meta.dims)) {
    return false;
  }

  std::set<int> visited_idx;
  axis->resize(meta.strides.size());
  *src_shape = meta.dims;
  *src_stride = meta.strides;
  for (int i = 0; i < meta.strides.size(); ++i) {
    int64_t max_num = 0;
    int max_idx = -1;
    for (int j = 0; j < meta.strides.size(); ++j) {
      if (visited_idx.count(j)) {
        continue;
      }
      if (meta.strides[j] < 1) {
        return false;
      }
      if (meta.strides[j] > max_num) {
        max_num = meta.strides[j];
        max_idx = j;
      }
    }
    if (max_idx == -1) {
      return false;
    }
    if (i != 0 && (*src_stride)[i - 1] == max_num && (*src_shape)[i - 1] != 1 &&
        meta.dims[max_idx] != 1) {
      return false;
    }
    visited_idx.insert(max_idx);
    (*src_stride)[i] = max_num;
    (*src_shape)[i] = meta.dims[max_idx];
    (*axis)[max_idx] = i;
  }

  return DenseTensorMeta::calc_strides(*src_shape) == *src_stride;
}

inline void CanonicalizePureTransposeView(const DenseTensor& input,
                                          bool* transpose,
                                          DenseTensor* output) {
  *output = input;
  if (input.meta().is_contiguous()) {
    return;
  }

  DDim src_shape;
  DDim src_stride;
  std::vector<int> axis;
  if (!IsOnlyTransposedTensor(input, &src_shape, &src_stride, &axis)) {
    return;
  }

  const auto trans_dims = axis.size();
  if (trans_dims < 2 || axis[trans_dims - 1] != trans_dims - 2 ||
      axis[trans_dims - 2] != trans_dims - 1) {
    return;
  }

  auto meta = output->meta();
  meta.dims = src_shape;
  meta.strides = src_stride;
  output->set_meta(meta);
  *transpose = !*transpose;
}

template <typename T, typename Context>
void MatmulGradStrideKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& y,
                            const DenseTensor& out_grad,
                            bool transpose_x,
                            bool transpose_y,
                            DenseTensor* dx,
                            DenseTensor* dy) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }

  DenseTensor x_ = x;
  DenseTensor y_ = y;
  DenseTensor out_grad_ = out_grad;

  CanonicalizePureTransposeView(x, &transpose_x, &x_);
  CanonicalizePureTransposeView(y, &transpose_y, &y_);

  if (!x_.meta().is_contiguous()) {
    x_ = Tensor2Contiguous<Context>(dev_ctx, x_);
  }
  if (!y_.meta().is_contiguous()) {
    y_ = Tensor2Contiguous<Context>(dev_ctx, y_);
  }
  if (!out_grad_.meta().is_contiguous()) {
    out_grad_ = Tensor2Contiguous<Context>(dev_ctx, out_grad_);
  }

  PrepareStridedOut(dx);
  PrepareStridedOut(dy);

  phi::MatmulGradKernel<T, Context>(
      dev_ctx, x_, y_, out_grad_, transpose_x, transpose_y, dx, dy);
}

}  // namespace phi

PD_REGISTER_KERNEL(matmul_grad,
                   GPU,
                   STRIDED,
                   phi::MatmulGradStrideKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

#endif
