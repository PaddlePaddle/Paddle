/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/detail/strided_memcpy.h"
namespace paddle {
namespace operators {

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
inline void StridedMemcpy(const platform::DeviceContext& dev_ctx, const T* src,
                          const framework::DDim& src_stride,
                          const framework::DDim& dst_dim,
                          const framework::DDim& dst_stride, T* dst) {
  paddle::operators::detail::StridedCopyDimVisitor<T> func(
      dev_ctx, src, src_stride, dst_stride, dst);
  dst_dim.apply_visitor(func);
}

// Strided numel memory copy from src to dst by the specified axis
//
// For example, for a tensor dims [4, 20, 100], the strieded numel is
// [8000, 2000, 100]
//
// NOTE: The src and dst tensor should have the same elements
// except the specified axis.
template <typename T>
inline void StridedNumelCopyWithAxis(const platform::DeviceContext& ctx,
                                     int64_t axis, T* dst,
                                     const framework::DDim& dst_stride_numel,
                                     const T* src,
                                     const framework::DDim& src_stride_numel,
                                     int64_t size) {
  int64_t before = dst_stride_numel[0] / dst_stride_numel[axis];
  int64_t src_after = src_stride_numel[axis];
  int64_t dst_after = dst_stride_numel[axis];
  auto place = ctx.GetPlace();

  PADDLE_ENFORCE_EQ(src_stride_numel.size(), dst_stride_numel.size(),
                    "src and dst tensor should have the same dims size.");

  for (int64_t i = 0; i < axis; ++i) {
    if (i < axis) {
      PADDLE_ENFORCE_EQ(src_stride_numel[i] / src_stride_numel[axis],
                        dst_stride_numel[i] / dst_stride_numel[axis],
                        "src and dst should have the same elements "
                        "except the specified axis.");
    } else if (i == axis) {
      continue;
    } else {
      PADDLE_ENFORCE_EQ(src_stride_numel[i], dst_stride_numel[i],
                        "src and dst should have the same elements "
                        "except the specified axis.");
    }
  }
  VLOG(1) << "StridedNumelCopyWithAxis CUDA Copy before: " << before
          << " dst_after: " << dst_after << " src_after: " << src_after;
  for (int64_t i = 0; i < before; ++i) {
    if (platform::is_cpu_place(place)) {
      VLOG(1) << "StridedNumelCopyWithAxis CPU Copy Begin";
      auto& cpu_place = BOOST_GET_CONST(platform::CPUPlace, place);
      memory::Copy(cpu_place, dst + i * dst_after, cpu_place,
                   src + i * src_after, sizeof(T) * size);
    } else {
#ifdef PADDLE_WITH_CUDA
      VLOG(1) << "StridedNumelCopyWithAxis CUDA Copy Begin";
      VLOG(1) << "StridedNumelCopyWithAxis CUDA Copy i: " << i;
      auto& gpu_place = BOOST_GET_CONST(platform::CUDAPlace, place);
      auto& cuda_ctx =
          reinterpret_cast<const platform::CUDADeviceContext&>(ctx);

      memory::Copy(gpu_place, dst + i * dst_after, gpu_place,
                   src + i * src_after, sizeof(T) * size, cuda_ctx.stream());
#else
      PADDLE_THROW("Paddle is not compiled with GPU");
#endif
    }
  }
}

template <typename T>
inline void StridedNumelCopyWithAxis(platform::CUDADeviceContext* dst_ctx,
                                     const platform::DeviceContext& src_ctx,
                                     int64_t axis, T* dst,
                                     const framework::DDim& dst_stride_numel,
                                     const T* src,
                                     const framework::DDim& src_stride_numel,
                                     int64_t size) {
  int64_t before = dst_stride_numel[0] / dst_stride_numel[axis];
  int64_t src_after = src_stride_numel[axis];
  int64_t dst_after = dst_stride_numel[axis];

  auto dst_place = dst_ctx->GetPlace();
  auto src_place = src_ctx.GetPlace();
  VLOG(1) << "StridedNumelCopyWithAxis CPU<->CUDA Copy before: " << before
          << " dst_after: " << dst_after << " src_after: " << src_after;
  PADDLE_ENFORCE_EQ(src_stride_numel.size(), dst_stride_numel.size(),
                    "src and dst tensor should have the same dims size.");

  for (int64_t i = 0; i < axis; ++i) {
    if (i < axis) {
      PADDLE_ENFORCE_EQ(src_stride_numel[i] / src_stride_numel[axis],
                        dst_stride_numel[i] / dst_stride_numel[axis],
                        "src and dst should have the same elements "
                        "except the specified axis.");
    } else if (i == axis) {
      continue;
    } else {
      PADDLE_ENFORCE_EQ(src_stride_numel[i], dst_stride_numel[i],
                        "src and dst should have the same elements "
                        "except the specified axis.");
    }
  }

  for (int64_t i = 0; i < before; ++i) {
    VLOG(1) << "StridedNumelCopyWithAxis CPU<->GPU Copy";
    auto& cpu_place = BOOST_GET_CONST(platform::CPUPlace, src_place);
    auto& gpu_place = BOOST_GET_CONST(platform::CUDAPlace, dst_place);
    memory::Copy(gpu_place, dst + i * dst_after, cpu_place, src + i * src_after,
                 sizeof(T) * size, dst_ctx->stream());
  }
}

template <typename T>
inline void StridedMemcpyWithAxis0(
    const platform::DeviceContext& dev_ctx, const framework::Tensor& input,
    const std::vector<const framework::Tensor*>& shape_refer,
    std::vector<framework::Tensor*>* outputs) {
  const framework::DDim in_stride = stride_numel(input.dims());
  const int axis = 0;
  size_t input_offset = 0;

  for (size_t i = 0; i < outputs->size(); ++i) {
    auto out_stride = stride_numel(shape_refer[i]->dims());
    auto out = outputs->at(i);
    if (out != nullptr) {
      StridedNumelCopyWithAxis<T>(dev_ctx, axis, out->data<T>(), out_stride,
                                  input.data<T>() + input_offset, in_stride,
                                  out_stride[axis]);
    }
    input_offset += out_stride[axis];
  }
}

}  // namespace operators
}  // namespace paddle
