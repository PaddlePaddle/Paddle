// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/fake_quantize_functor.h"
#include "paddle/utils/optional.h"

namespace phi {

template <typename T, typename Context>
void MovingAverageAbsMaxScaleKernel(
    const Context &dev_ctx,
    const DenseTensor &x,
    const paddle::optional<DenseTensor> &in_accum_in,
    const paddle::optional<DenseTensor> &in_state_in,
    float moving_rate,
    bool is_test,
    DenseTensor *out,
    DenseTensor *out_scale,
    DenseTensor *out_state,
    DenseTensor *out_accum) {
  auto *in = &x;

  if (out != nullptr) {
    dev_ctx.template Alloc<T>(out);
    phi::Copy(dev_ctx, *in, dev_ctx.GetPlace(), false, out);
  }

  // testing
  if (is_test) {
    return;
  }

  // training
  auto *in_accum = in_accum_in.get_ptr();
  auto *in_state = in_state_in.get_ptr();
  phi::DenseTensor tmp_scale;
  tmp_scale.Resize(common::make_dim(1));
  T *cur_scale_data = dev_ctx.template Alloc<T>(&tmp_scale);

  phi::funcs::FindAbsMaxFunctor<Context, T>()(
      dev_ctx, in->data<T>(), in->numel(), cur_scale_data);

  dev_ctx.template Alloc<T>(out_state);
  dev_ctx.template Alloc<T>(out_accum);
  dev_ctx.template Alloc<T>(out_scale);

  phi::funcs::FindMovingAverageAbsMaxFunctor<Context, T>()(dev_ctx,
                                                           *in_accum,
                                                           *in_state,
                                                           cur_scale_data,
                                                           moving_rate,
                                                           out_state,
                                                           out_accum,
                                                           out_scale);
}
}  // namespace phi
