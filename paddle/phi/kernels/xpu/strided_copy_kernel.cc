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

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/strided_copy_kernel.h"
namespace phi {

template <typename T, typename Context>
void StridedCopyKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& out_stride,
                       int64_t offset,
                       DenseTensor* out) {
  DenseTensorMeta meta = input.meta();
  meta.strides = make_ddim(out_stride);
  meta.dims = make_ddim(dims);
  meta.offset = offset;
  out->set_meta(meta);

  PADDLE_ENFORCE_EQ(input.numel(),
                    out->numel(),
                    common::errors::InvalidArgument(
                        "Input numel(%d) must be equal with out numel(%d).",
                        input.numel(),
                        out->numel()));

  const auto numel = input.numel();
  if (numel <= 0) {
    return;
  }

  PADDLE_ENFORCE_NOT_NULL(out->data<T>(),
                          common::errors::InvalidArgument(
                              "StridedCopyKernel's out tensor must complete "
                              "mutable data before call kernel."));

  // use XPUCopyTypeTrait to deal with double and int16_t copy instead of
  // XPUTypeTrait
  using XPUType = typename XPUCopyTypeTrait<T>::Type;

  int r = 0;
  auto input_data = reinterpret_cast<const XPUType*>(input.data<T>());
  auto output_data = reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out));
  PADDLE_ENFORCE_NOT_NULL(output_data,
                          common::errors::InvalidArgument(
                              "StridedCopyKernel's out tensor must complete "
                              "mutable data before call kernel."));
  if (input.numel() == 1) {
    r = xpu::copy<XPUType>(dev_ctx.x_context(), input_data, output_data, 1);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
  } else {
    const int64_t in_rank = input.dims().size();
    const int64_t out_rank = out->dims().size();
    PADDLE_ENFORCE_GE(
        in_rank,
        1,
        common::errors::InvalidArgument(
            "XPU StridedCopyKernel requires input rank in [1, 9], but got "
            "input rank = %d.",
            in_rank));
    PADDLE_ENFORCE_LE(
        in_rank,
        9,
        common::errors::InvalidArgument(
            "XPU StridedCopyKernel requires input rank in [1, 9], but got "
            "input rank = %d.",
            in_rank));
    PADDLE_ENFORCE_GE(
        out_rank,
        1,
        common::errors::InvalidArgument(
            "XPU StridedCopyKernel requires output rank in [1, 9], but got "
            "output rank = %d.",
            out_rank));
    PADDLE_ENFORCE_LE(
        out_rank,
        9,
        common::errors::InvalidArgument(
            "XPU StridedCopyKernel requires output rank in [1, 9], but got "
            "output rank = %d.",
            out_rank));

    const int64_t data_bytes_in = static_cast<int64_t>(input.Holder()->size()) -
                                  static_cast<int64_t>(input.meta().offset);
    const int64_t data_bytes_out = static_cast<int64_t>(out->Holder()->size()) -
                                   static_cast<int64_t>(out->meta().offset);
    const int64_t data_elems_in = std::max<int64_t>(0, data_bytes_in) /
                                  static_cast<int64_t>(sizeof(XPUType));
    const int64_t data_elems_out = std::max<int64_t>(0, data_bytes_out) /
                                   static_cast<int64_t>(sizeof(XPUType));
    const int64_t data_size = std::max(data_elems_in, data_elems_out);
    r = xpu::strided_copy<XPUType>(dev_ctx.x_context(),
                                   input_data,
                                   output_data,
                                   data_size,
                                   vectorize<int64_t>(input.dims()),
                                   vectorize<int64_t>(out->dims()),
                                   vectorize<int64_t>(input.strides()),
                                   vectorize<int64_t>(out->strides()));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "strided_copy");
  }
}

#ifdef PADDLE_WITH_XPU_FFT
template <typename T>
typename std::enable_if<std::is_same<T, phi::complex64>::value ||
                        std::is_same<T, phi::complex128>::value>::type
StridedCopyKernelImpl(const XPUContext& dev_ctx,
                      const DenseTensor& input,
                      const std::vector<int64_t>& dims,
                      const std::vector<int64_t>& out_stride,
                      int64_t offset,
                      DenseTensor* out) {
  DenseTensorMeta meta = input.meta();
  meta.strides = common::make_ddim(out_stride);
  meta.dims = common::make_ddim(dims);
  meta.offset = offset;
  out->set_meta(meta);

  PADDLE_ENFORCE_EQ(input.numel(),
                    out->numel(),
                    common::errors::InvalidArgument(
                        "Input numel(%d) must be equal with out numel(%d).",
                        input.numel(),
                        out->numel()));

  const auto numel = input.numel();
  if (numel <= 0) {
    return;
  }

  PADDLE_ENFORCE_NOT_NULL(out->data<T>(),
                          common::errors::InvalidArgument(
                              "StridedCopyKernel's out tensor must complete "
                              "mutable data before call kernel."));
  auto* out_data = dev_ctx.template Alloc<T>(out);
  PADDLE_ENFORCE_NOT_NULL(out_data,
                          common::errors::InvalidArgument(
                              "StridedCopyKernel's out tensor must complete "
                              "mutable data before call kernel."));

  using CopyT = float;
  const auto* in_ptr = reinterpret_cast<const CopyT*>(input.data<T>());
  auto* out_ptr = reinterpret_cast<CopyT*>(out_data);

  constexpr const char* kTypeName =
      std::is_same<T, phi::complex64>::value ? "complex64" : "complex128";
  PADDLE_ENFORCE_EQ(
      reinterpret_cast<uintptr_t>(in_ptr) % alignof(CopyT),
      0UL,
      common::errors::PreconditionNotMet(
          "XPU StridedCopyKernel for %s requires %d-byte aligned input "
          "pointer.",
          kTypeName,
          static_cast<int>(alignof(CopyT))));
  PADDLE_ENFORCE_EQ(
      reinterpret_cast<uintptr_t>(out_ptr) % alignof(CopyT),
      0UL,
      common::errors::PreconditionNotMet(
          "XPU StridedCopyKernel for %s requires %d-byte aligned output "
          "pointer.",
          kTypeName,
          static_cast<int>(alignof(CopyT))));

  constexpr int64_t kCopyUnitsPerElem = sizeof(T) / sizeof(CopyT);
  int r = 0;

  if (numel == 1) {
    r = xpu::copy<CopyT>(
        dev_ctx.x_context(), in_ptr, out_ptr, kCopyUnitsPerElem);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    return;
  }

  const int64_t in_rank = input.dims().size();
  const int64_t out_rank = out->dims().size();
  PADDLE_ENFORCE_GE(
      in_rank,
      1,
      common::errors::InvalidArgument(
          "XPU StridedCopyKernel for %s requires input rank in [1, 8], but got "
          "input rank = %d.",
          kTypeName,
          in_rank));
  PADDLE_ENFORCE_LE(
      in_rank,
      8,
      common::errors::InvalidArgument(
          "XPU StridedCopyKernel for %s requires input rank in [1, 8], but got "
          "input rank = %d.",
          kTypeName,
          in_rank));
  PADDLE_ENFORCE_GE(
      out_rank,
      1,
      common::errors::InvalidArgument("XPU StridedCopyKernel for %s requires "
                                      "output rank in [1, 8], but got "
                                      "output rank = %d.",
                                      kTypeName,
                                      out_rank));
  PADDLE_ENFORCE_LE(
      out_rank,
      8,
      common::errors::InvalidArgument("XPU StridedCopyKernel for %s requires "
                                      "output rank in [1, 8], but got "
                                      "output rank = %d.",
                                      kTypeName,
                                      out_rank));

  auto in_strides_vec = common::vectorize<int64_t>(input.strides());
  auto out_strides_vec = common::vectorize<int64_t>(out->strides());
  if (kCopyUnitsPerElem > 1) {
    for (auto& s : in_strides_vec) {
      s *= kCopyUnitsPerElem;
    }
    for (auto& s : out_strides_vec) {
      s *= kCopyUnitsPerElem;
    }
  }

  auto in_dims_vec = common::vectorize<int64_t>(input.dims());
  auto out_dims_vec = common::vectorize<int64_t>(out->dims());
  in_dims_vec.push_back(kCopyUnitsPerElem);
  out_dims_vec.push_back(kCopyUnitsPerElem);
  in_strides_vec.push_back(1);
  out_strides_vec.push_back(1);
  const int64_t data_bytes_in = static_cast<int64_t>(input.Holder()->size()) -
                                static_cast<int64_t>(input.meta().offset);
  const int64_t data_bytes_out = static_cast<int64_t>(out->Holder()->size()) -
                                 static_cast<int64_t>(out->meta().offset);
  const int64_t data_elems_in =
      std::max<int64_t>(0, data_bytes_in) / static_cast<int64_t>(sizeof(CopyT));
  const int64_t data_elems_out = std::max<int64_t>(0, data_bytes_out) /
                                 static_cast<int64_t>(sizeof(CopyT));
  const int64_t data_size = std::max(data_elems_in, data_elems_out);
  r = xpu::strided_copy<CopyT>(dev_ctx.x_context(),
                               in_ptr,
                               out_ptr,
                               data_size,
                               in_dims_vec,
                               out_dims_vec,
                               in_strides_vec,
                               out_strides_vec);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "strided_copy");
}

template <>
void StridedCopyKernel<phi::complex64, XPUContext>(
    const XPUContext& dev_ctx,
    const DenseTensor& input,
    const std::vector<int64_t>& dims,
    const std::vector<int64_t>& out_stride,
    int64_t offset,
    DenseTensor* out) {
  StridedCopyKernelImpl<phi::complex64>(
      dev_ctx, input, dims, out_stride, offset, out);
}
template <>
void StridedCopyKernel<phi::complex128, XPUContext>(
    const XPUContext& dev_ctx,
    const DenseTensor& input,
    const std::vector<int64_t>& dims,
    const std::vector<int64_t>& out_stride,
    int64_t offset,
    DenseTensor* out) {
  StridedCopyKernelImpl<phi::complex128>(
      dev_ctx, input, dims, out_stride, offset, out);
}
#endif

}  // namespace phi

PD_REGISTER_KERNEL(strided_copy,
                   XPU,
                   ALL_LAYOUT,
                   phi::StridedCopyKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   float,
                   double,
#ifdef PADDLE_WITH_XPU_FFT
                   phi::complex64,
                   phi::complex128,
#endif
                   phi::float16,
                   phi::bfloat16) {
}
