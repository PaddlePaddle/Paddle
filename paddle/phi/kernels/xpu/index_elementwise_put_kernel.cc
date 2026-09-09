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

#include "paddle/phi/kernels/index_elementwise_put_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/index_elementwise.h"
#include "paddle/phi/kernels/funcs/index_elementwise_utils.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"

namespace phi {

template <typename T, typename Context, typename IndexT = int>
void XPUIndexElementwisePutWithTensorKernel(
    const Context& dev_ctx,
    const DenseTensor& input,
    const DenseTensor& value,
    const std::vector<const DenseTensor*>& index,
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& input_strides,
    const std::vector<int64_t>& index_dims,
    const std::vector<int64_t>& index_strides,
    const int64_t slice_offset,
    DenseTensor* output) {
  int64_t numel = 0;
  bool is_initialized = output->initialized();
  bool is_same_place = true;
  if (is_initialized) {
    is_same_place = (input.place() == output->place());
  }
  if (!is_initialized || !is_same_place) {
    Copy(dev_ctx, input, dev_ctx.GetPlace(), false, output);
  }

  if (index.empty()) {
    return;
  }

  int64_t num_indices = 0;
  std::vector<int64_t> shape_tmp;
  std::vector<int64_t> stride_tmp;
  funcs::cal_shape_stride(index_dims, &num_indices, &shape_tmp, &stride_tmp);

  // xpu::index_elementwise_tensor can only walk the x view forwards, so rewrite
  // its reversed axes. The scatter then consumes `value` reversed along
  // `flip_axes`; since the destination is written in place it is `value` that
  // has to be flipped, not the output.
  const std::vector<int64_t> value_dims = vectorize<int64_t>(value.dims());
  std::vector<int64_t> x_strides = input_strides;
  int64_t x_offset = slice_offset;
  std::vector<int64_t> flip_axes;
  funcs::NormalizeNegativeStrides(input_dims,
                                  phi::SizeOf(input.dtype()),
                                  &x_strides,
                                  &x_offset,
                                  &flip_axes);
  // `value` is right-aligned against `input_dims` by compute_strides(), and a
  // broadcast axis carries stride 0 so its traversal order does not matter.
  std::vector<int64_t> value_flip_axes;
  const int64_t rank_diff = static_cast<int64_t>(input_dims.size()) -
                            static_cast<int64_t>(value_dims.size());
  for (int64_t axis : flip_axes) {
    const int64_t value_axis = axis - rank_diff;
    if (value_axis >= 0 && value_dims[value_axis] > 1) {
      value_flip_axes.push_back(value_axis);
    }
  }

  auto sizes = std::array<int64_t, 25>{};
  auto strides = std::array<int64_t, 25>{};
  for (int64_t i = 0; i < num_indices; i++) {
    sizes[i] = index_dims[i];
    strides[i] = index_strides[i];
  }
  std::array<int64_t*, 3> strides_array;
  std::vector<int64_t> desired_shape;
  std::array<std::vector<int64_t>, 3> strides_vec;
  funcs::IndexPutStride<3>(input_dims,
                           x_strides,
                           phi::SizeOf(input.dtype()),
                           value_dims,
                           vectorize<int64_t>(value.strides()),
                           phi::SizeOf(value.dtype()),
                           shape_tmp,
                           stride_tmp,
                           phi::SizeOf(index[0]->dtype()),
                           &desired_shape,
                           &strides_array,
                           &numel,
                           strides_vec);
  if (numel == 0 || funcs::HasEmptyIndex(index)) {
    return;
  }
  const int64_t N = numel;
  PADDLE_ENFORCE_EQ(true,
                    (N >= 0 && N <= std::numeric_limits<int32_t>::max()),
                    common::errors::PreconditionNotMet(
                        "the value of N should be in [0, "
                        "std::numeric_limits<int32_t>::max()]"));

  dev_ctx.template Alloc<T>(output);
  using XPUType = typename XPUTypeTrait<T>::Type;
  using XPUTypeIndexT = typename XPUTypeTrait<IndexT>::Type;

  // passed vector params for XPU
  std::vector<const XPUTypeIndexT*> index_ptrs_vec;
  std::vector<int64_t> index_numel_vec;
  for (int i = 0; i < num_indices; i++) {
    // since XPU WRAPPER_CHECK_PTR only supports original GM ptrs, so we pass
    // the IndexT* type ptrs, which is different from the CPU/GPU's char* ptr.
    index_ptrs_vec.push_back(
        reinterpret_cast<const XPUTypeIndexT*>(index[i]->data<IndexT>()));
    // index_numel_vec is for the length of WRAPPER_CHECK_PTR
    index_numel_vec.push_back(index[i]->numel());
  }
  std::vector<int64_t> sizes_vec =
      std::vector<int64_t>(sizes.begin(), sizes.begin() + num_indices);
  std::vector<int64_t> orig_strides_vec =
      std::vector<int64_t>(strides.begin(), strides.begin() + num_indices);
  std::vector<std::vector<int64_t>> strides_vec_vec =
      std::vector<std::vector<int64_t>>(strides_vec.begin(), strides_vec.end());

  const char* in_ptr = reinterpret_cast<const char*>(value.data<T>());
  char* out_ptr = reinterpret_cast<char*>(output->data<T>()) + x_offset;

  // for checkptr and checksum in XPU
  int64_t data_size_in = value.Holder()->size() - value.meta().offset;
  int64_t data_size_out = output->Holder()->size() - output->meta().offset;

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  if (!value_flip_axes.empty()) {
    using XPUCopyType = typename XPUCopyTypeTrait<T>::Type;
    auto* flipped = RAII_GUARD.alloc_l3_or_gm<XPUCopyType>(value.numel());
    int rf =
        xpu::flip<XPUCopyType>(dev_ctx.x_context(),
                               reinterpret_cast<const XPUCopyType*>(in_ptr),
                               flipped,
                               value_dims,
                               value_flip_axes);
    PADDLE_ENFORCE_XDNN_SUCCESS(rf, "flip");
    in_ptr = reinterpret_cast<const char*>(flipped);
    data_size_in = value.numel() * static_cast<int64_t>(sizeof(T));
  }

  bool is_get = false;
  int r = xpu::index_elementwise_tensor<XPUType, XPUTypeIndexT>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(in_ptr),  // XPU ptr
      reinterpret_cast<XPUType*>(out_ptr),       // XPU ptr
      index_ptrs_vec,                            // vec of XPU ptrs
      input_dims,                                // CPU vec
      index_numel_vec,                           // CPU vec
      desired_shape,                             // CPU vec
      sizes_vec,                                 // CPU vec
      orig_strides_vec,                          // CPU vec
      strides_vec_vec,                           // CPU vec
      N,                                         // int64_t
      data_size_in,                              // int64_t
      data_size_out,                             // int64_t
      is_get);                                   // true for get, false for put
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "index_elementwise_tensor_put");
}

template <typename T, typename Context, typename IndexT = int>
void XPUIndexElementwisePutKernel(const Context& dev_ctx,
                                  const DenseTensor& input,
                                  const Scalar& value,
                                  const std::vector<const DenseTensor*>& index,
                                  const std::vector<int64_t>& input_dims,
                                  const std::vector<int64_t>& input_strides,
                                  const std::vector<int64_t>& index_dims,
                                  const std::vector<int64_t>& index_strides,
                                  const int64_t slice_offset,
                                  DenseTensor* output) {
  int64_t numel = 0;
  bool is_initialized = output->initialized();
  bool is_same_place = true;
  if (is_initialized) {
    is_same_place = (input.place() == output->place());
  }
  if (!is_initialized || !is_same_place) {
    Copy(dev_ctx, input, dev_ctx.GetPlace(), false, output);
  }

  if (index.empty()) {
    return;
  }

  int64_t num_indices = 0;
  std::vector<int64_t> shape_tmp;
  std::vector<int64_t> stride_tmp;
  funcs::cal_shape_stride(index_dims, &num_indices, &shape_tmp, &stride_tmp);

  // xpu::index_elementwise_scalar can only walk the x view forwards. There is
  // no dense operand here, so reversing the traversal order of an axis needs no
  // compensation: every visited position gets the same scalar.
  std::vector<int64_t> x_strides = input_strides;
  int64_t x_offset = slice_offset;
  std::vector<int64_t> flip_axes;
  funcs::NormalizeNegativeStrides(input_dims,
                                  phi::SizeOf(input.dtype()),
                                  &x_strides,
                                  &x_offset,
                                  &flip_axes);

  auto sizes = std::array<int64_t, DDim::kMaxRank + 1>{};
  auto strides = std::array<int64_t, DDim::kMaxRank + 1>{};
  for (int64_t i = 0; i < num_indices; i++) {
    sizes[i] = index_dims[i];
    strides[i] = index_strides[i];
  }
  std::array<int64_t*, 3> strides_array;
  std::vector<int64_t> desired_shape;
  std::array<std::vector<int64_t>, 3> strides_vec;
  funcs::IndexPutStride<3>(input_dims,
                           x_strides,
                           phi::SizeOf(input.dtype()),
                           {},
                           {},
                           4,
                           shape_tmp,
                           stride_tmp,
                           phi::SizeOf(index[0]->dtype()),
                           &desired_shape,
                           &strides_array,
                           &numel,
                           strides_vec);
  if (numel == 0 || funcs::HasEmptyIndex(index)) {
    return;
  }
  const int64_t N = numel;
  PADDLE_ENFORCE_EQ(true,
                    (N >= 0 && N <= std::numeric_limits<int32_t>::max()),
                    common::errors::PreconditionNotMet(
                        "the value of N should be in [0, "
                        "std::numeric_limits<int32_t>::max()]"));

  dev_ctx.template Alloc<T>(output);
  using XPUType = typename XPUTypeTrait<T>::Type;
  using XPUTypeIndexT = typename XPUTypeTrait<IndexT>::Type;

  // passed vector params for XPU
  std::vector<const XPUTypeIndexT*> index_ptrs_vec;
  std::vector<int64_t> index_numel_vec;
  for (int i = 0; i < std::min(num_indices, (int64_t)index.size()); i++) {
    // since XPU WRAPPER_CHECK_PTR only supports original GM ptrs, so we pass
    // the IndexT* type ptrs, which is different from the CPU/GPU's char* ptr.
    index_ptrs_vec.push_back(
        reinterpret_cast<const XPUTypeIndexT*>(index[i]->data<IndexT>()));
    // index_numel_vec is for the length of WRAPPER_CHECK_PTR
    index_numel_vec.push_back(index[i]->numel());
  }
  std::vector<int64_t> sizes_vec =
      std::vector<int64_t>(sizes.begin(), sizes.begin() + num_indices);
  std::vector<int64_t> orig_strides_vec =
      std::vector<int64_t>(strides.begin(), strides.begin() + num_indices);
  std::vector<std::vector<int64_t>> strides_vec_vec =
      std::vector<std::vector<int64_t>>(strides_vec.begin(), strides_vec.end());

  char* out_ptr = reinterpret_cast<char*>(output->data<T>()) + x_offset;

  // for checkptr and checksum in XPU
  int64_t data_size_out = output->Holder()->size() - output->meta().offset;

  const XPUType value_T = static_cast<XPUType>(value.to<T>());
  bool is_get = false;

  // bool and int64_t index will be handled in XPU's op wrapper
  int r = xpu::index_elementwise_scalar<XPUType, XPUTypeIndexT>(
      dev_ctx.x_context(),
      reinterpret_cast<XPUType*>(out_ptr),  // XPU ptr
      value_T,                              // scalar
      index_ptrs_vec,                       // vec of XPU ptrs
      input_dims,                           // CPU vec
      index_numel_vec,                      // CPU vec
      desired_shape,                        // CPU vec
      sizes_vec,                            // CPU vec
      orig_strides_vec,                     // CPU vec
      strides_vec_vec,                      // CPU vec
      N,                                    // int64_t
      data_size_out,                        // int64_t
      is_get);                              // false for put
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "index_elementwise_scalar_put");
}

template <typename T, typename Context>
void IndexElementwisePutWithTensorKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const std::vector<const DenseTensor*>& index,
    const DenseTensor& value,
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& input_strides,
    const std::vector<int64_t>& index_dims,
    const std::vector<int64_t>& index_strides,
    const int64_t slice_offset,
    DenseTensor* out) {
  if (index.empty()) {
    if (out->numel() == 0) {
      dev_ctx.template Alloc<T>(out);
      return;
    }
    bool is_initialized = out->initialized();
    bool is_same_place = is_initialized ? (x.place() == out->place()) : false;
    dev_ctx.template Alloc<T>(out);
    if (!is_initialized || !is_same_place) {
      Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    }
    return;
  }
  const auto& index_type = index[0]->dtype();
  PADDLE_ENFORCE_EQ(index_type == DataType::INT64,
                    true,
                    common::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s], but "
                        "desires to be [%s].",
                        index_type,
                        DataType::INT64));
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  if (out->numel() == 0) return;
  XPUIndexElementwisePutWithTensorKernel<T, Context, int64_t>(dev_ctx,
                                                              x,
                                                              value,
                                                              index,
                                                              input_dims,
                                                              input_strides,
                                                              index_dims,
                                                              index_strides,
                                                              slice_offset,
                                                              out);
}

template <typename T, typename Context>
void IndexElementwisePutKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const std::vector<const DenseTensor*>& index,
                               const Scalar& value,
                               const std::vector<int64_t>& input_dims,
                               const std::vector<int64_t>& input_strides,
                               const std::vector<int64_t>& index_dims,
                               const std::vector<int64_t>& index_strides,
                               const int64_t slice_offset,
                               DenseTensor* out) {
  if (index.empty()) {
    if (out->numel() == 0) {
      dev_ctx.template Alloc<T>(out);
      return;
    }
    bool is_initialized = out->initialized();
    bool is_same_place = is_initialized ? (x.place() == out->place()) : false;
    dev_ctx.template Alloc<T>(out);
    if (!is_initialized || !is_same_place) {
      Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    }
    return;
  }
  const auto& index_type = index[0]->dtype();
  PADDLE_ENFORCE_EQ(index_type == DataType::INT64 ||
                        (index_type == DataType::BOOL && index.size() == 1),
                    true,
                    common::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s], but "
                        "desires to be [%s].",
                        index_type,
                        DataType::INT64));
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  if (out->numel() == 0) return;
  XPUIndexElementwisePutKernel<T, Context, int64_t>(dev_ctx,
                                                    x,
                                                    value,
                                                    index,
                                                    input_dims,
                                                    input_strides,
                                                    index_dims,
                                                    index_strides,
                                                    slice_offset,
                                                    out);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_elementwise_put,
                   XPU,
                   ALL_LAYOUT,
                   phi::IndexElementwisePutKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16) {}

PD_REGISTER_KERNEL(index_elementwise_put_with_tensor,
                   XPU,
                   ALL_LAYOUT,
                   phi::IndexElementwisePutWithTensorKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16) {}
