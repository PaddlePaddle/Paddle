/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/tensor.h"

namespace paddle {
namespace framework {

/**
 * @brief   Copy the content of external tensor to a new place.
 *
 * @param[in] src        The external tensor.
 * @param[in] dst_place  The dst place.
 * @param[in] ctx        The device context contains device resources.
 *
 * @note    CopyFrom supports CPU <-> GPU, GPU <-> GPU.
 */
void CopyFrom(const Tensor& src, const platform::Place& dst_place,
              const platform::DeviceContext& ctx, Tensor* dst);

/**
 * @brief   Copy the content of an external vector to a tensor.
 *
 * @param[in] src        The external tensor.
 * @param[in] ctx        The device context contains device resources.
 *
 * * @note    CopyFromVector assumes that the tensor has been resized
 *            before invoking.
 */
template <typename T>
void CopyFromVector(const std::vector<T>& src,
                    const platform::DeviceContext& ctx, Tensor* dst);

/**
 * @brief   Copy the content of a tensor to a vector
 *
 * @param[in] src        The external tensor.
 * @param[in] ctx        The device context contains device resources.
 *
 * * @note    CopyFromVector assumes that the tensor has been resized
 *            before invoking.
 */
template <typename T>
void CopyToVector(const Tensor& src, const platform::DeviceContext& ctx,
                  std::vector<T>* dst);

void CopyFrom(const Tensor& src, const platform::Place& dst_place,
              const platform::DeviceContext& ctx, Tensor* dst) {
  src.check_memory_size();

  dst->Resize(src.dims());
  auto src_place = src.place();
  auto src_ptr = src.data<void>();

  auto dst_ptr = dst->mutable_data(dst_place, src.type());

  auto size = src.numel() * SizeOfType(src.type());

  if (platform::is_cpu_place(src_place) && platform::is_cpu_place(dst_place)) {
    memory::Copy(boost::get<platform::CPUPlace>(dst_place), dst_ptr,
                 boost::get<platform::CPUPlace>(src_place), src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(src_place) &&  // NOLINT
           platform::is_cpu_place(dst_place)) {
    auto src_gpu_place = boost::get<platform::GPUPlace>(src_place);
    auto dst_cpu_place = boost::get<platform::CPUPlace>(dst_place);
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx_place));
    auto ctx_gpu_place = boost::get<platform::GPUPlace>(ctx_place);
    PADDLE_ENFORCE_EQ(src_gpu_place, ctx_gpu_place);
    memory::Copy(
        dst_cpu_place, dst_ptr, src_gpu_place, src_ptr, size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  } else if (platform::is_cpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    auto src_cpu_place = boost::get<platform::CPUPlace>(src_place);
    auto dst_gpu_place = boost::get<platform::GPUPlace>(dst_place);
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx_place));
    auto ctx_gpu_place = boost::get<platform::GPUPlace>(ctx_place);
    PADDLE_ENFORCE_EQ(dst_gpu_place, ctx_gpu_place);
    memory::Copy(
        dst_gpu_place, dst_ptr, src_cpu_place, src_ptr, size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  } else if (platform::is_gpu_place(src_place) &&
             platform::is_gpu_place(dst_place)) {
    auto src_gpu_place = boost::get<platform::GPUPlace>(src_place);
    auto dst_gpu_place = boost::get<platform::GPUPlace>(dst_place);
    auto ctx_place = ctx.GetPlace();
    PADDLE_ENFORCE(platform::is_gpu_place(ctx_place));
    auto ctx_gpu_place = boost::get<platform::GPUPlace>(ctx_place);
    PADDLE_ENFORCE_EQ(src_gpu_place, ctx_gpu_place);
    memory::Copy(
        dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  }
#endif
}

template <typename T>
void CopyFromVector(const std::vector<T>& src,
                    const platform::DeviceContext& ctx, Tensor* dst) {
  auto dst_place = ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  platform::CPUPlace src_place;
  auto dst_ptr = static_cast<void*>(dst->mutable_data<T>(dst_place));
  auto size = src.size() * sizeof(T);

  if (platform::is_cpu_place(dst_place)) {
    memory::Copy(boost::get<platform::CPUPlace>(dst_place), dst_ptr, src_place,
                 src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(dst_place)) {  // NOLINT
    memory::Copy(
        boost::get<platform::GPUPlace>(dst_place), dst_ptr, src_place, src_ptr,
        size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  }
#endif
}

template <typename T>
void CopyToVector(const Tensor& src, const platform::DeviceContext& ctx,
                  std::vector<T>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<T>());
  platform::CPUPlace src_place = boost::get<platform::CPUPlace>(src.place());
  auto size = src.numel() * sizeof(T);

  auto dst_place = ctx.GetPlace();
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  if (platform::is_cpu_place(src_place)) {
    memory::Copy(boost::get<platform::CPUPlace>(dst_place), dst_ptr, src_place,
                 src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(src_place)) {  // NOLINT
    memory::Copy(
        boost::get<platform::GPUPlace>(dst_place), dst_ptr, src_place, src_ptr,
        size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  }
#endif
}

}  // namespace framework
}  // namespace paddle
