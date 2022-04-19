/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <functional>
#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/cusparse.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"

namespace paddle {
namespace platform {

class CusparseHandleHolder {
 public:
  explicit CusparseHandleHolder(cudaStream_t stream) {
// ROCM is not yet supported
#if defined(PADDLE_WITH_CUDA)
// The generic APIs is supported from CUDA10.1
#if CUDA_VERSION >= 10010
    PADDLE_RETRY_CUDA_SUCCESS(dynload::cusparseCreate(&handle_));
    PADDLE_RETRY_CUDA_SUCCESS(dynload::cusparseSetStream(handle_, stream));
#endif
#endif
  }
  const cusparseHandle_t& GetCusparseHandle() const { return handle_; }

  ~CusparseHandleHolder() PADDLE_MAY_THROW {
#if defined(PADDLE_WITH_CUDA)
#if CUDA_VERSION >= 10010
    PADDLE_RETRY_CUDA_SUCCESS(dynload::cusparseDestroy(handle_));
#endif
#endif
  }

  inline void Call(
      const std::function<void(phi::sparseHandle_t)>& callback) const {
    std::lock_guard<std::mutex> guard(mtx_);
    callback(handle_);
  }

 private:
  DISABLE_COPY_AND_ASSIGN(CusparseHandleHolder);

  cusparseHandle_t handle_;
  mutable std::mutex mtx_;
};

}  // namespace platform
}  // namespace paddle
