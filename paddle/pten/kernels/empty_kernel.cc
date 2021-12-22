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

#include "paddle/pten/kernels/empty_kernel.h"

#include "paddle/pten/backends/all_context.h"
#include "paddle/pten/core/kernel_registry.h"

namespace pten {

template <typename T, typename ContextT>
void Empty(const ContextT& dev_ctx,
           const ScalarArray& shape,
           DenseTensor* out) {
  out->Resize(paddle::framework::make_ddim(shape.GetData()));
}

template <typename T, typename ContextT>
void EmptyLike(const ContextT& dev_ctx, DenseTensor* out) {
  out->mutable_data<T>();
}

}  // namespace pten

PT_REGISTER_CTX_KERNEL(empty,
                       CPU,
                       ALL_LAYOUT,
                       pten::Empty,
                       bool,
                       int,
                       int64_t,
                       float,
                       double,
                       paddle::platform::float16) {}

PT_REGISTER_CTX_KERNEL(empty_like,
                       CPU,
                       ALL_LAYOUT,
                       pten::EmptyLike,
                       bool,
                       int,
                       int64_t,
                       float,
                       double,
                       paddle::platform::float16) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_REGISTER_CTX_KERNEL(empty,
                       GPU,
                       ALL_LAYOUT,
                       pten::Empty,
                       bool,
                       int,
                       int64_t,
                       float,
                       double,
                       paddle::platform::float16) {}

PT_REGISTER_CTX_KERNEL(empty_like,
                       GPU,
                       ALL_LAYOUT,
                       pten::EmptyLike,
                       bool,
                       int,
                       int64_t,
                       float,
                       double,
                       paddle::platform::float16) {}
#endif
