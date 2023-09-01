// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

namespace phi {
namespace funcs {

template <typename T, typename func_t>
struct BinaryFunctor {
  __device__ T operator()(T a, T b) const { return f(a, b); }
  // NB: scalar is stored in higher precision!
  BinaryFunctor(func_t f_) : f(f_) {}

 private:
  func_t f;
};

template <typename T, typename Func>
__global__ void binary_kernel(
    const T* x, const T* y, const int num, Func f, T* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    out[idx] = f(x[idx], y[idx]);
  }
}

// template <typename func_t>
// void test_func( const func_t& f)
// {
//   using traits  = FunctionTraits<func_t>;
//   // using ArgsT = typename traits::ArgsTuple;
//   using arg1_t = typename traits::template arg<0>::type;
//   arg1_t t(0.0);
//   f( t );
// }

template <typename T, typename Func>
void binary(const DenseTensor& x,
            const DenseTensor& y,
            DenseTensor* out,
            Func func) {
  int numel = phi::product(x.dims());
  int thread = 256;
  int block_size = ((numel - 1) / 256) + 1;
  binary_kernel<<<block_size, thread>>>(
      x.data<T>(), y.data<T>(), numel, func, out->data<T>());
}

}  // namespace funcs
}  // namespace phi
