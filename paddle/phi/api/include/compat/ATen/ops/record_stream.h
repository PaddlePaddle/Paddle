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

#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAStream.h>
#endif

namespace at {
inline void Tensor::record_stream(at::Stream s) const {
  auto dense_tensor =
      std::dynamic_pointer_cast<phi::DenseTensor>(tensor_.impl());
  PD_CHECK(dense_tensor != nullptr,
           "record_stream only supports DenseTensor, but got a non-dense "
           "tensor implementation.");
  PD_CHECK(dense_tensor->place().GetType() != phi::AllocationType::CPU,
           "record_stream is not supported for CPU tensors.");
#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)) && \
    !defined(PADDLE_WITH_CUSTOM_DEVICE)
  paddle::memory::RecordStream(
      dense_tensor->Holder(), reinterpret_cast<gpuStream_t>(s.native_handle()));
#elif defined(PADDLE_WITH_XPU)
  paddle::memory::RecordStream(dense_tensor->Holder(),
                               reinterpret_cast<XPUStream>(s.native_handle()));
#elif defined(PADDLE_WITH_CUSTOM_DEVICE)
  paddle::memory::RecordStream(
      dense_tensor->Holder(),
      reinterpret_cast<phi::stream::stream_t>(s.native_handle()));
#else
  (void)s;
  (void)dense_tensor;
  PD_THROW(
      "record_stream is not supported: no GPU/XPU/Custom device enabled "
      "in this build.");
#endif
}

}  // namespace at
