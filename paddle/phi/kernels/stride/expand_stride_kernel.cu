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

#include "paddle/phi/kernels/expand_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/stride/elementwise_stride_base.cu.h"

COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_stride_compute_kernel);

namespace phi {

template <typename T, typename Context>
void ExpandStrideKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const IntArray& shape,
                        DenseTensor* out) {
  bool invalid_stride = false;
  if (x.numel() <= 0 || !x.IsInitialized() || x.dims().size() > 7) {
    invalid_stride = true;
  }
  if (out->numel() <= 0 || out->dims().size() > 7) {
    invalid_stride = true;
  }

  DenseTensor x_;
  if (!FLAGS_use_stride_compute_kernel || invalid_stride) {
    if (!x.meta().is_contiguous()) {
      x_ = Tensor2Contiguous<Context>(dev_ctx, x);
    } else {
      x_ = x;
    }

    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
    phi::ExpandKernel<T, Context>(dev_ctx, x_, shape, out);
    return;
  }

  if (!FLAGS_use_stride_compute_kernel) {
    PADDLE_THROW(
        common::errors::Fatal("FLAGS_use_stride_compute_kernel is closed. "
                              "Kernel using DenseTensorIterator "
                              "be called, something wrong has happened!"));
  }

  auto in_dims = x.dims();
  auto expand_shape = shape.GetData();
  if (expand_shape.empty()) {
    *out = x;
    return;
  }
  auto vec_in_dims = common::vectorize<int64_t>(in_dims);
  auto diff = expand_shape.size() - vec_in_dims.size();
  PADDLE_ENFORCE_GE(
      diff,
      0,
      common::errors::InvalidArgument(
          "The rank of the target shape (%d) must be greater than or equal to "
          "the rank of the input tensor (%d).",
          expand_shape.size(),
          vec_in_dims.size()));
  vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  auto out_shape = vec_in_dims;
  bool has_zero_dim = false;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    if (i < diff) {
      PADDLE_ENFORCE_GE(
          expand_shape[i],
          0,
          common::errors::InvalidArgument(
              "The expanded size (%d) for non-existing dimensions must be "
              "positive for expand_v2 op.",
              expand_shape[i]));
      if (expand_shape[i] == 0) has_zero_dim = true;
      out_shape[i] = expand_shape[i];
    } else if (expand_shape[i] == -1) {
      out_shape[i] = vec_in_dims[i];
    } else if (expand_shape[i] == 0) {
      PADDLE_ENFORCE_EQ(
          vec_in_dims[i] == 1 || vec_in_dims[i] == expand_shape[i],
          true,
          common::errors::InvalidArgument(
              "The %d-th dimension of input tensor (%d) must match or be "
              "broadcastable to the corresponding dimension (%d) in shape.",
              i,
              vec_in_dims[i],
              expand_shape[i]));
      out_shape[i] = 0;
      has_zero_dim = true;
    } else if (expand_shape[i] > 0) {
      PADDLE_ENFORCE_EQ(
          vec_in_dims[i] == 1 || vec_in_dims[i] == expand_shape[i],
          true,
          common::errors::InvalidArgument(
              "The %d-th dimension of input tensor (%d) must match or be "
              "broadcastable to the corresponding dimension (%d) in shape.",
              i,
              vec_in_dims[i],
              expand_shape[i]));
      out_shape[i] = expand_shape[i];
    }
  }

  if (has_zero_dim) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  std::vector<int64_t> out_dims;
  std::vector<int64_t> out_strides;

  int64_t ndim = static_cast<int64_t>(expand_shape.size());
  int64_t tensor_dim = static_cast<int64_t>(x.dims().size());

  std::vector<int64_t> expandedSizes(ndim, 0);
  std::vector<int64_t> expandedStrides(ndim, 0);

  for (int64_t i = ndim - 1; i >= 0; --i) {
    int64_t offset = ndim - 1 - i;
    int64_t dim = tensor_dim - 1 - offset;
    int64_t size = (dim >= 0) ? x.dims()[dim] : 1;
    int64_t stride = (dim >= 0) ? x.strides()[dim]
                                : expandedSizes[i + 1] * expandedStrides[i + 1];
    int64_t targetSize = expand_shape[i];
    if (targetSize == -1) {
      targetSize = size;
    }
    if (size != targetSize) {
      size = targetSize;
      stride = 0;
    }
    expandedSizes[i] = size;
    expandedStrides[i] = stride;
  }

  auto meta = out->meta();
  meta.dims =
      DDim(expandedSizes.data(), static_cast<int>(expandedSizes.size()));
  meta.strides =
      DDim(expandedStrides.data(), static_cast<int>(expandedStrides.size()));

  out->set_meta(meta);
  out->ResetHolder(x.Holder());
  out->ShareInplaceVersionCounterWith(x);
}

}  // namespace phi

PD_REGISTER_KERNEL(expand,
                   GPU,
                   STRIDED,
                   phi::ExpandStrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::float8_e4m3fn,
                   phi::float8_e5m2,
                   phi::complex64,
                   phi::complex128) {}
