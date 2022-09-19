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

#include "paddle/phi/kernels/selected_rows/add_n_kernel.h"

#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace sr {
template <typename T, typename Context>
void AddNKernel(const Context &dev_ctx,
                const std::vector<const SelectedRows *> &x,
                SelectedRows *out) {
  dev_ctx.template Alloc<T>(out->mutable_value());

  bool in_place = false;
  if (x.size() > 0 && x[0]->value().Holder() == out->value().Holder()) {
    in_place = true;
  }

  if (in_place && x.size() < 2) {
    return;
  }

  std::vector<const phi::SelectedRows *> inputs;
  SelectedRows temp_in0;

  if (in_place) {
    auto &in0 = *x[0];
    temp_in0.set_height(in0.height());
    temp_in0.set_rows(in0.rows());
    Copy<Context>(
        dev_ctx, in0.value(), in0.place(), false, temp_in0.mutable_value());
    inputs.push_back(&temp_in0);
    for (size_t i = 1; i < x.size(); ++i) {
      auto &in = *x[i];
      if (in.rows().size() > 0) {
        inputs.push_back(&in);
      }
    }
  } else {
    for (auto in_var : x) {
      auto &in = *in_var;
      if (in.rows().size() > 0) {
        inputs.push_back(in_var);
      }
    }
  }

  out->mutable_rows()->clear();

  bool has_data = false;
  for (auto &in : inputs) {
    if (in->rows().size() > 0) {
      has_data = true;
      break;
    }
  }
  if (has_data) {
    paddle::operators::math::scatter::MergeAdd<Context, T> merge_add;
    merge_add(dev_ctx, inputs, out);

    out->SyncIndex();

  } else {
    // no data, just set a empty out tensor.
    auto *out_dense = out->mutable_value();
    out_dense->clear();
    out_dense->Resize(phi::make_ddim({0}));
    dev_ctx.template Alloc<T>(out_dense);
  }
}
}  // namespace sr
}  // namespace phi
