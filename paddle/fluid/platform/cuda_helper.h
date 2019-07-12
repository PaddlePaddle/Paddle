// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/cublas.h"
#include "paddle/fluid/platform/macros.h"

#if CUDA_VERSION < 9000
enum cublasMath_t { CUBLAS_DEFAULT_MATH = 0 };
#endif

namespace paddle {
namespace platform {

class CublasHandleHolder {
 public:
  CublasHandleHolder(cudaStream_t stream, cublasMath_t math_type) {
    PADDLE_ENFORCE(dynload::cublasCreate(&handle_));
    PADDLE_ENFORCE(dynload::cublasSetStream(handle_, stream));
#if CUDA_VERSION >= 9000
    if (math_type == CUBLAS_TENSOR_OP_MATH) {
      PADDLE_ENFORCE(
          dynload::cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));
    }
#endif
  }

  ~CublasHandleHolder() { PADDLE_ENFORCE(dynload::cublasDestroy(handle_)); }

  template <typename Callback>
  inline void Call(Callback &&callback) const {
    std::lock_guard<std::mutex> guard(mtx_);
    callback(handle_);
  }

 private:
  DISABLE_COPY_AND_ASSIGN(CublasHandleHolder);

  cublasHandle_t handle_;
  mutable std::mutex mtx_;
};

}  // namespace platform
}  // namespace paddle
