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
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/device_context.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/gpu_context.h"
#endif

namespace phi {
namespace funcs {
namespace detail {

template <typename T, int Rank>
struct StridedMemcpyFunctor;

template <typename T>
struct StridedMemcpyFunctor<T, 0> {
  void operator()(const phi::DeviceContext& dev_ctx,
                  const T* src,
                  const int64_t* src_stride UNUSED,
                  const int64_t* dst_dim UNUSED,
                  const int64_t* dst_stride UNUSED,
                  T* dst) const {
    auto place = dev_ctx.GetPlace();
    if (place.GetType() == phi::AllocationType::CPU) {
      auto& cpu_place = place;
      memory_utils::Copy(cpu_place, dst, cpu_place, src, sizeof(T));
    } else {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      auto& gpu_place = place;
      auto& cuda_ctx = reinterpret_cast<const phi::GPUContext&>(dev_ctx);
      memory_utils::Copy(
          gpu_place, dst, gpu_place, src, sizeof(T), cuda_ctx.stream());
#else
      PADDLE_THROW(
          phi::errors::Unavailable("Paddle is not compiled with GPU."));
#endif
    }
  }
};

template <typename T>
struct StridedMemcpyFunctor<T, 1> {
  void operator()(const phi::DeviceContext& dev_ctx,
                  const T* src,
                  const int64_t* src_stride UNUSED,
                  const int64_t* dst_dim,
                  const int64_t* dst_stride UNUSED,
                  T* dst) const {
    auto place = dev_ctx.GetPlace();
    if (place.GetType() == phi::AllocationType::CPU) {
      auto& cpu_place = place;
      memory_utils::Copy(
          cpu_place, dst, cpu_place, src, sizeof(T) * dst_dim[0]);
    } else {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      auto& gpu_place = place;
      auto& cuda_ctx = reinterpret_cast<const phi::GPUContext&>(dev_ctx);
      memory_utils::Copy(gpu_place,
                         dst,
                         gpu_place,
                         src,
                         sizeof(T) * dst_dim[0],
                         cuda_ctx.stream());
#else
      PADDLE_THROW(
          phi::errors::Unavailable("Paddle is not compiled with GPU."));
#endif
    }
  }
};

template <typename T, int Rank>
struct StridedMemcpyFunctor {
  void operator()(const phi::DeviceContext& dev_ctx,
                  const T* src,
                  const int64_t* src_stride,
                  const int64_t* dst_dim,
                  const int64_t* dst_stride,
                  T* dst) const {
    for (int64_t i = 0; i < dst_dim[0]; ++i) {
      StridedMemcpyFunctor<T, Rank - 1> func;
      func(dev_ctx, src, src_stride + 1, dst_dim + 1, dst_stride + 1, dst);
      src += src_stride[0];
      dst += dst_stride[0];
    }
  }
};

template <typename T>
struct StridedCopyDimVisitor {
  StridedCopyDimVisitor(const phi::DeviceContext& dev_ctx,
                        const T* src,
                        const phi::DDim& src_stride,
                        const phi::DDim& dst_stride,
                        T* dst)
      : dev_ctx_(dev_ctx),
        src_(src),
        src_stride_(src_stride),
        dst_stride_(dst_stride),
        dst_(dst) {}

  template <int D>
  void operator()(const phi::Dim<D>& dst_dim) const {
    StridedMemcpyFunctor<T, D> functor;
    functor(dev_ctx_,
            src_,
            src_stride_.Get(),
            dst_dim.Get(),
            dst_stride_.Get(),
            dst_);
  }

  const phi::DeviceContext& dev_ctx_;
  const T* src_;
  const phi::DDim& src_stride_;
  const phi::DDim& dst_stride_;
  T* dst_;
};

}  // namespace detail
}  // namespace funcs
}  // namespace phi
