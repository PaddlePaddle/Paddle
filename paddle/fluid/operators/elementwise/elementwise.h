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
#ifndef PADDLE_FLUID_OPERATORS_ELEMENTWISE_ELEMENTWISE_H_
#define PADDLE_FLUID_OPERATORS_ELEMENTWISE_ELEMENTWISE_H_

#include <cuda.h>
#include <cuda_fp16.h>
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
namespace paddle {
namespace operators {
template <typename Functor>
struct CommonSameDimsElemwise {
  inline void operator()(const framework::ExecutionContext& ctx,
                         const framework::Tensor* x, const framework::Tensor* y,
                         framework::Tensor* z) const {
    auto size = x->numel();
    dim3 gird_size = dim3((size / 2 + TILE_SIZE - 1) / TILE_SIZE, 1);
    dim3 block_size = dim3(TILE_SIZE, 1);
    const half* x2 =
        reinterpret_cast<const half*>(x->data<platform::float16>());
    const half* y2 =
        reinterpret_cast<const half*>(y->data<platform::float16>());
    half* z2 = reinterpret_cast<half*>(z->data<platform::float16>());
    Functor functor;
    functor(x2, y2, z2, size, ctx, gird_size, block_size);
  }
};
}  // namespace operators
}  // namespace paddle
#endif  // PADDLE_FLUID_OPERATORS_ELEMENTWISE_ELEMENTWISE_H_
