// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/stack_functor.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include <thrust/device_vector.h>
#endif

namespace phi {

template <typename T, typename Context>
void UnStackKernel(const Context &dev_ctx,
                   const DenseTensor &x,
                   int axis,
                   int num,
                   std::vector<DenseTensor *> outs) {
  auto *dy = &x;
  auto dx = outs;
  if (axis < 0) axis += dy->dims().size();

  int n = dy->dims()[axis];
  std::vector<T *> dx_datas(n);  // NOLINT
  for (int i = 0; i < n; i++) {
    dx_datas[i] = dev_ctx.template Alloc<T>(dx[i]);
  }
  auto dy_data = dy->data<T>();
  if (dy->numel() == 0) return;
  int pre = 1;
  for (int i = 0; i < axis; ++i) pre *= dy->dims()[i];
  int total_num = dy->numel();
  int post = total_num / (n * pre);

#if defined(__NVCC__) || defined(__HIPCC__)
  thrust::device_vector<T *> device_dx_vec(dx_datas);
  auto dx_data_arr = device_dx_vec.data().get();
#else
  auto dx_data_arr = dx_datas.data();
#endif
  phi::funcs::StackGradFunctorForRange(
      dev_ctx, dx_data_arr, dy_data, total_num, n, post);
#if defined(__NVCC__) || defined(__HIPCC__)
  // Wait() must be called because device_dx_vec may be destructed before
  // kernel ends
  dev_ctx.Wait();
#endif
}

}  // namespace phi
