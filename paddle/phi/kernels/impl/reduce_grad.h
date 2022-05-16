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

#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/reduce_grad_functions.h"

namespace phi {

template <typename Context,
          typename T,
          typename Functor,
          bool kNoNeedBufferX = false,
          bool kNoNeedBufferY = false>
void ComputeFromInput(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& out_grad,
                      const paddle::optional<DenseTensor>& out,
                      const DenseTensor& input2,
                      const std::vector<int64_t>& dims,
                      bool keep_dim,
                      bool reduce_all,
                      DenseTensor* x_grad) {
  auto* input0 = &x;
  auto* input1 = out.get_ptr();
  auto* output = x_grad;
  dev_ctx.template Alloc<T>(output);

  // The dims has full dim, set the reduce_all is True
  const auto& input_dim_size = x.dims().size();
  std::set<int> dims_set(dims.begin(), dims.end());
  bool full_dim = true;
  for (auto i = 0; i < input_dim_size; i++) {
    if (dims_set.find(i) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  reduce_all = (reduce_all || full_dim);
  // NOTE: EigenTensor::From() uses tensor->data()
  // if op has NoNeedBufferVarsInferer, the corresponding kNoNeedBufferX or
  // kNoNeedBufferY should set true
  // and use fake var that has same dims.
  if (kNoNeedBufferX) {
    input0 = output;
  }
  if (kNoNeedBufferY) {
    input1 = &input2;
  }

  const std::vector<int> const_dims{dims.begin(), dims.end()};

  // NOTE(dengkaipeng): Out is unnecessary in some reduce kernel and
  // not be set as Input in grad Maker, use Out_grad to replace here
  if (!input1) input1 = &input2;
  Functor functor;

  funcs::LaunchReduceGradKernel<Context, T, Functor>(dev_ctx,
                                                     input0,
                                                     input1,
                                                     &input2,
                                                     output,
                                                     functor,
                                                     const_dims,
                                                     reduce_all);
}

template <typename Context,
          typename T,
          typename Functor,
          bool kNoNeedBufferX = false,
          bool kNoNeedBufferY = false>
void ReduceGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const paddle::optional<DenseTensor>& out,
                      const DenseTensor& out_grad,
                      const std::vector<int64_t>& dims,
                      bool keep_dim,
                      bool reduce_all,
                      DenseTensor* x_grad) {
  if (x.dtype() != out_grad.dtype()) {
    DenseTensorMeta x_grad_meta(
        out_grad.dtype(), x_grad->dims(), x_grad->layout());
    DenseTensor x_grad_tmp =
        phi::Empty<Context>(dev_ctx, std::move(x_grad_meta));
    ComputeFromInput<Context, T, Functor, kNoNeedBufferX, kNoNeedBufferY>(
        dev_ctx,
        x,
        out_grad,
        out,
        out_grad,
        dims,
        keep_dim,
        reduce_all,
        &x_grad_tmp);

    phi::CastKernel<T>(dev_ctx, x_grad_tmp, x.dtype(), x_grad);
  } else {
    ComputeFromInput<Context, T, Functor, kNoNeedBufferX, kNoNeedBufferY>(
        dev_ctx,
        x,
        out_grad,
        out,
        out_grad,
        dims,
        keep_dim,
        reduce_all,
        x_grad);
  }
}

}  // namespace phi
