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

#include "paddle/common/macros.h"
#include "paddle/phi/backends/dynload/rocblas.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace platform {

/*
 * Summary: Grid stride looping macro in CUDA kernel
 *
 *  [ Why need this macro? ]
 *
 *    The original looping in CUDA kernel is:
 *
 *    `for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
 *        i += blockDim.x * gridDim.x)`
 *
 *    This for condition is risky. The value of `blockIdx.x * blockDim.x`
 *    may be large, such as over 1GB, the first iteration is no problem here,
 *    but when `i += blockDim.x * gridDim.x` is executed, the value of i
 *    will greater than INT_MAX and overflow becomes negative value, at
 *    this time, the cycle condition `i < (n)` is still satisfied, so it
 *    will cause illegal access to cuda memory.
 *
 *    Here is a real example in ERINE, it will trigger above error.
 *    The related data are:
 *      - blockIdx.x = 2172938
 *      - blockDim.x = 512
 *      - blockIdx.x * blockDim.x = 1112543864
 *      - INT_MAX = 2147483647
 *
 *    So we polish the for condition as follow, the int64_t __index__ will
 *    prevent overflow in the loop increment.
 *
 * Parameters:
 *    - i: loop index
 *    - num: total element numbers
 *
 * Examples:
 *    template <typename T>
 *    __global__ void Scale(T* logit_grad, const T* loss_grad, const int num,
 *                      const int d, const int remain) {
 *    CUDA_KERNEL_LOOP(index, num) {
 *      int idx_n = index / d;
 *      int idx_remain = index % remain;
 *      logit_grad[index] *= loss_grad[idx_n * remain + idx_remain];
 *      }
 *    }
 *
 */

#define CUDA_KERNEL_LOOP_TYPE(i, num, index_type)                           \
  int64_t __index__ =                                                       \
      static_cast<int64_t>(hipBlockIdx_x) * hipBlockDim_x + hipThreadIdx_x; \
  int64_t __stride__ = static_cast<int64_t>(hipBlockDim_x) * hipGridDim_x;  \
  for (index_type i = __index__; __index__ < (num);                         \
       __index__ += __stride__, i = __index__)

class CublasHandleHolder {
 public:
  explicit CublasHandleHolder(hipStream_t stream) {
    PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::rocblas_create_handle(&handle_));
    PADDLE_RETRY_CUDA_SUCCESS(
        phi::dynload::rocblas_set_stream(handle_, stream));
  }

  const rocblas_handle& GetCublasHandle() const { return handle_; }

  ~CublasHandleHolder() PADDLE_MAY_THROW {
    PADDLE_RETRY_CUDA_SUCCESS(phi::dynload::rocblas_destroy_handle(handle_));
  }

  template <typename Callback>
  inline void Call(Callback&& callback) const {
    std::lock_guard<std::mutex> guard(mtx_);
    callback(handle_);
  }

 private:
  DISABLE_COPY_AND_ASSIGN(CublasHandleHolder);

  rocblas_handle handle_;
  mutable std::mutex mtx_;
};

}  // namespace platform
}  // namespace paddle
