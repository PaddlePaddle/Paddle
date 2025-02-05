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
#include "paddle/phi/kernels/view_kernel.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/strided_reshape_utils.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

template <typename Context>
void ViewShapeStridedKernel(const Context& dev_ctx,
                            const DenseTensor& input,
                            const std::vector<int64_t>& dims,
                            DenseTensor* out) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  // infer dims
  auto infer_dim = -1;
  auto new_size = 1;
  auto numel = input.numel();
  std::vector<int64_t> dims_copy = dims;
  for (int dim = 0, ndim = dims_copy.size(); dim < ndim; ++dim) {
    if (dims_copy[dim] == -1) {
      if (infer_dim >= 0) {
        PADDLE_THROW(
            common::errors::Fatal("Only one dimension can be inferred"));
      }
      infer_dim = dim;
    } else if (dims_copy[dim] >= 0) {
      new_size *= dims_copy[dim];
    } else {
      PADDLE_THROW(common::errors::OutOfRange("Tensor idx is out of range"));
    }
  }
  PADDLE_ENFORCE_NE(new_size,
                    0,
                    common::errors::Unavailable(
                        "cannot reshape tensor of 0 elements into shape "));
  if (infer_dim >= 0 && new_size > 0 && numel % new_size == 0) {
    dims_copy[infer_dim] = numel / new_size;
  }

  DDim new_dims = DDim(dims_copy.data(), static_cast<int>(dims_copy.size()));
  DDim stride;
  if (ReshapeStride(input.dims(), input.strides(), new_dims, stride)) {
    auto meta = input.meta();
    meta.dims = new_dims;
    meta.strides = stride;
    meta.offset = input.offset();
    out->set_meta(meta);
    out->ResetHolder(input.Holder());
    out->ShareInplaceVersionCounterWith(input);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The Tensor can not be viewed, please call reshape."));
  }
}

template <typename Context>
void ViewDtypeKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     DataType dtype,
                     DenseTensor* out) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  size_t input_dtype_size = phi::SizeOf(input.dtype());
  size_t output_dtype_size = phi::SizeOf(dtype);

  if (input_dtype_size == output_dtype_size) {
    auto meta = input.meta();
    meta.dtype = dtype;
    out->set_meta(meta);
    out->ResetHolder(input.Holder());
    out->ShareInplaceVersionCounterWith(input);
  } else if (input_dtype_size == 0) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The Tensor's shape is [] can not be viewed."));
  } else if (input_dtype_size > output_dtype_size) {
    PADDLE_ENFORCE_EQ(
        input.strides()[input.strides().size() - 1],
        1,
        common::errors::InvalidArgument(
            "input.strides[-1] must be 1 to view %s as %s, but got %d",
            input.dtype(),
            dtype,
            input.strides()[input.strides().size() - 1]));
    size_t times = input_dtype_size / output_dtype_size;  // NOLINT

    DDim output_dims = input.dims();
    output_dims[output_dims.size() - 1] =
        output_dims[output_dims.size() - 1] * times;  // NOLINT

    DDim output_stride = input.strides();
    for (int i = 0; i < output_stride.size(); i++) {
      output_stride[i] = output_stride[i] * times;  // NOLINT
    }
    output_stride[output_stride.size() - 1] = 1;

    auto meta = input.meta();
    meta.dtype = dtype;
    meta.dims = output_dims;
    meta.strides = output_stride;
    meta.offset = input.offset() * times;
    out->set_meta(meta);
    out->ResetHolder(input.Holder());
    out->ShareInplaceVersionCounterWith(input);
  } else {
    PADDLE_ENFORCE_EQ(
        input.strides()[input.strides().size() - 1],
        1,
        common::errors::InvalidArgument(
            "input.strides[%d] must be 1 to view %s as %s, but got %d",
            input.strides().size() - 1,
            input.dtype(),
            dtype,
            input.strides()[input.strides().size() - 1]));
    size_t times = input_dtype_size / output_dtype_size;
    PADDLE_ENFORCE_EQ(
        input.dims()[input.dims().size() - 1] % times,
        0,
        common::errors::InvalidArgument(
            "input.shape[%d](%d) must be multiple of %d to view %s as %s",
            input.dims().size() - 1,
            input.dims()[input.dims().size() - 1],
            times,
            input.dtype(),
            dtype));
    PADDLE_ENFORCE_EQ(
        input.offset() % times,
        0,
        common::errors::InvalidArgument(
            "input.offset(%d) must be multiple of %d to view %s as %s",
            input.offset(),
            times,
            input.dtype(),
            dtype));

    DDim output_dims = input.dims();
    output_dims[output_dims.size() - 1] =
        output_dims[output_dims.size() - 1] / times;  // NOLINT

    DDim output_stride = input.strides();
    for (int i = 0; i < output_stride.size(); i++) {
      PADDLE_ENFORCE_EQ(
          output_stride[i] % times,
          0,
          common::errors::InvalidArgument("input.strides[%d](%d) must be be "
                                          "multiple of %d to view %s as %s",
                                          i,
                                          output_stride[i],
                                          times,
                                          input.dtype(),
                                          dtype));
      output_stride[i] = output_stride[i] / times;  // NOLINT
    }
    output_stride[output_stride.size() - 1] = 1;

    auto meta = input.meta();
    meta.dtype = dtype;
    meta.dims = output_dims;
    meta.strides = output_stride;
    meta.offset = input.offset() / times;
    out->set_meta(meta);
    out->ResetHolder(input.Holder());
    out->ShareInplaceVersionCounterWith(input);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(view_shape,
                                         STRIDED,
                                         phi::ViewShapeStridedKernel) {}

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(view_dtype,
                                         STRIDED,
                                         phi::ViewDtypeKernel) {}
