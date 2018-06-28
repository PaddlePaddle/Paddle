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
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {

void TensorCopy(const Tensor& src, const platform::Place& dst_place,
                const platform::DeviceContext& ctx, Tensor* dst);
void TensorCopy(const Tensor& src, const platform::Place& dst_place,
                Tensor* dst);
void TensorCopySync(const Tensor& src, const platform::Place& dst_place,
                    Tensor* dst);

template <typename T>
void TensorFromVector(const std::vector<T>& src,
                      const platform::DeviceContext& ctx, Tensor* dst);
template <typename T>
void TensorFromVector(const std::vector<T>& src, Tensor* dst);

template <typename T>
void TensorToVector(const Tensor& src, const platform::DeviceContext& ctx,
                    std::vector<T>* dst);
template <typename T>
void TesnorToVector(const Tensor& src, std::vector<T>* dst);

bool TensorContainsNAN(const framework::Tensor& tensor);
bool TensorContainsInf(const framework::Tensor& tensor);

void TensorToStream(std::ostream& os, const Tensor& tensor,
                    const platform::DeviceContext& dev_ctx);
void TensorFromStream(std::istream& is, Tensor* tensor,
                      const platform::DeviceContext& dev_ctx);

//
// The implementation of template functions.
//

template <typename T>
void TensorFromVector(const std::vector<T>& src,
                      const platform::DeviceContext& ctx, Tensor* dst) {
  auto dst_place = ctx.GetPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  platform::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<T>(dst_place));
  auto size = src.size() * sizeof(T);

  if (platform::is_cpu_place(dst_place)) {
    memory::Copy(boost::get<platform::CPUPlace>(dst_place), dst_ptr, src_place,
                 src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(dst_place)) {  // NOLINT
    memory::Copy(
        boost::get<platform::CUDAPlace>(dst_place), dst_ptr, src_place, src_ptr,
        size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  }
#endif
}

template <typename T>
void TensorFromVector(const std::vector<T>& src, Tensor* dst) {
  platform::CPUPlace dst_place = platform::CPUPlace();
  auto src_ptr = static_cast<const void*>(src.data());
  platform::CPUPlace src_place;
  dst->Resize({static_cast<int64_t>(src.size())});
  auto dst_ptr = static_cast<void*>(dst->mutable_data<T>(dst_place));
  auto size = src.size() * sizeof(T);

  memory::Copy(dst_place, dst_ptr, src_place, src_ptr, size);
}

template <typename T>
void TensorToVector(const Tensor& src, const platform::DeviceContext& ctx,
                    std::vector<T>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<T>());
  auto size = src.numel() * sizeof(T);

  platform::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  if (platform::is_cpu_place(src.place())) {
    memory::Copy(dst_place, dst_ptr,
                 boost::get<platform::CPUPlace>(src.place()), src_ptr, size);
  }
#ifdef PADDLE_WITH_CUDA
  else if (platform::is_gpu_place(src.place())) {  // NOLINT
    memory::Copy(
        dst_place, dst_ptr, boost::get<platform::CUDAPlace>(src.place()),
        src_ptr, size,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream());
  }
#endif
}

template <typename T>
void TensorToVector(const Tensor& src, std::vector<T>* dst) {
  auto src_ptr = static_cast<const void*>(src.data<T>());
  auto size = src.numel() * sizeof(T);

  platform::CPUPlace dst_place;
  dst->resize(src.numel());
  auto dst_ptr = static_cast<void*>(dst->data());

  PADDLE_ENFORCE(platform::is_cpu_place(src.place()));

  memory::Copy(dst_place, dst_ptr, boost::get<platform::CPUPlace>(src.place()),
               src_ptr, size);
}

}  // namespace framework
}  // namespace paddle
