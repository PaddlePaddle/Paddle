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
void UnStackGradKernel(const Context &dev_ctx,
                       const std::vector<const DenseTensor *> &x,
                       int axis,
                       DenseTensor *x_grad) {
  if (axis < 0) axis += (x[0]->dims().size() + 1);

  int n = static_cast<int>(x.size());
  auto *x_grad_data = dev_ctx.template Alloc<T>(x_grad);
  std::vector<const T *> x_datas(n);
  for (int i = 0; i < n; i++) x_datas[i] = x[i]->data<T>();

  int pre = 1;
  int post = 1;
  auto &dim = x[0]->dims();
  for (auto i = 0; i < axis; ++i) pre *= dim[i];
  for (auto i = axis; i < dim.size(); ++i) post *= dim[i];

#if defined(__NVCC__) || defined(__HIPCC__)
  int total_num = pre * n * post;

  thrust::device_vector<const T *> device_x_vec(x_datas);
  auto x_data_arr = device_x_vec.data().get();

  phi::funcs::StackFunctorForRange(
      dev_ctx, x_data_arr, x_grad_data, total_num, n, post);

  // Wait() must be called because device_x_vec may be destructed before
  // kernel ends
  dev_ctx.Wait();
#else
  auto x_data_arr = x_datas.data();

  size_t x_offset = 0;
  size_t y_offset = 0;
  for (int i = 0; i < pre; i++) {
    for (int j = 0; j < n; j++) {
      std::memcpy(
          x_grad_data + y_offset, x_data_arr[j] + x_offset, post * sizeof(T));
      y_offset += post;
    }
    x_offset += post;
  }
#endif
}

}  // namespace phi
