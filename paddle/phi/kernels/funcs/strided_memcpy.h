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

#pragma once
#include <vector>

#include "paddle/common/macros.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/detail/strided_memcpy.h"

namespace phi {
class CPUContext;
}  // namespace phi

namespace phi {
namespace funcs {

// Strided memory copy from src to dst.
//
// The src and dst should be both on dev_ctx.GetPlace(), otherwise, there will
// be a segment fault.
//
// The stride of an array (also referred to as increment, pitch or step size) is
// the number of locations in memory between beginnings of successive array
// elements
//
// For example, for tensor like [1, 3, 300, 300]. If there is no padding, the
// stride is [270000, 90000, 300, 1].
//
// NOTE: When use GPU, the memcpy is async. To sync memcpy, please invoke
// `dev_ctx.Wait()`.
template <typename T>
inline void StridedMemcpy(const phi::DeviceContext& dev_ctx,
                          const T* src,
                          const phi::DDim& src_stride,
                          const phi::DDim& dst_dim,
                          const phi::DDim& dst_stride,
                          T* dst) {
  detail::StridedCopyDimVisitor<T> func(
      dev_ctx, src, src_stride, dst_stride, dst);
  dst_dim.apply_visitor(func);
}

template <typename Context>
inline void CopyWithContext(const Context& ctx,
                            const Place& dst_place,
                            void* dst,
                            const Place& src_place,
                            const void* src,
                            size_t num) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_CUSTOM_DEVICE)
  memory_utils::Copy(dst_place, dst, src_place, src, num, ctx.stream());
#else
  PADDLE_THROW(
      common::errors::PreconditionNotMet("Paddle is not compiled with GPU."));
#endif
}

template <>
inline void CopyWithContext<phi::CPUContext>(const phi::CPUContext& ctx UNUSED,
                                             const Place& dst_place,
                                             void* dst,
                                             const Place& src_place,
                                             const void* src,
                                             size_t num) {
  memory_utils::Copy(dst_place, dst, src_place, src, num);
}

// Strided numel memory copy from src to dst by the specified axis
//
// For example, for a tensor dims [4, 20, 100], the strieded numel is
// [8000, 2000, 100]
//
// NOTE: The src and dst tensor should have the same elements
// except the specified axis.
template <typename T, typename Context>
inline void StridedNumelCopyWithAxis(const Context& ctx,
                                     int64_t axis,
                                     T* dst,
                                     const phi::DDim& dst_stride_numel,
                                     const T* src,
                                     const phi::DDim& src_stride_numel,
                                     int64_t size) {
  int64_t before = dst_stride_numel[0] / dst_stride_numel[axis];
  int64_t src_after = src_stride_numel[axis];
  int64_t dst_after = dst_stride_numel[axis];
  auto place = ctx.GetPlace();

  PADDLE_ENFORCE_EQ(src_stride_numel.size(),
                    dst_stride_numel.size(),
                    common::errors::InvalidArgument(
                        "Source and destination tensor should have the same "
                        "dimension size, but source tensor dimension size is "
                        "%u, destination tensor size is %u.",
                        src_stride_numel.size(),
                        dst_stride_numel.size()));

  for (int64_t i = 0; i < axis; ++i) {
    if (i < axis) {
      PADDLE_ENFORCE_EQ(
          src_stride_numel[i] / src_stride_numel[axis],
          dst_stride_numel[i] / dst_stride_numel[axis],
          common::errors::InvalidArgument(
              "Source and destination tensor should have the same number of "
              "elements except the specified axis, but the source elements "
              "number is %d, destination elements number is %d.",
              src_stride_numel[i] / src_stride_numel[axis],
              dst_stride_numel[i] / dst_stride_numel[axis]));
    } else if (i == axis) {
      continue;
    } else {
      PADDLE_ENFORCE_EQ(
          src_stride_numel[i],
          dst_stride_numel[i],
          common::errors::InvalidArgument(
              "Source and destination tensor should have the same number of "
              "elements except the specified axis, but the source elements "
              "number is %d, destination elements number is %d.",
              src_stride_numel[i],
              dst_stride_numel[i]));
    }
  }

  for (int64_t i = 0; i < before; ++i) {
    CopyWithContext<Context>(ctx,
                             place,
                             dst + i * dst_after,
                             place,
                             src + i * src_after,
                             sizeof(T) * size);
  }
}

template <typename T, typename Context>
inline void StridedMemcpyWithAxis0(
    const Context& dev_ctx,
    const phi::DenseTensor& input,
    const std::vector<const phi::DenseTensor*>& shape_refer,
    std::vector<phi::DenseTensor*>* outputs) {
  const phi::DDim in_stride = common::stride_numel(input.dims());
  const int axis = 0;
  size_t input_offset = 0;

  for (size_t i = 0; i < outputs->size(); ++i) {
    auto out_stride = common::stride_numel(shape_refer[i]->dims());
    auto out = outputs->at(i);
    if (out != nullptr && out->initialized() && out->numel() > 0) {
      StridedNumelCopyWithAxis<T, Context>(dev_ctx,
                                           axis,
                                           out->data<T>(),
                                           out_stride,
                                           input.data<T>() + input_offset,
                                           in_stride,
                                           out_stride[axis]);
    }
    input_offset += out_stride[axis];
  }
}

}  // namespace funcs
}  // namespace phi
