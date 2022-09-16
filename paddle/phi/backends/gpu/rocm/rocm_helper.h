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

namespace phi {
namespace backends {
namespace gpu {

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
  for (index_type i = __index__; __index__ < (num);                         \
       __index__ += hipBlockDim_x * hipGridDim_x, i = __index__)

}  // namespace gpu
}  // namespace backends
}  // namespace phi
