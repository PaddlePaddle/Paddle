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

#include "paddle/phi/kernels/add_n_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace phi {

template <typename T, typename Context>
void AddNArrayKernel(const Context& dev_ctx,
                     const std::vector<const TensorArray*>& x,
                     TensorArray* out) {
  for (auto& ele : *out) {
    dev_ctx.template Alloc<T>(&ele);
  }
  bool in_place = true;
  if (x.size() > 0 && x[0]->size() == out->size()) {
    for (size_t i = 0; i < out->size(); i++) {
      if (x[0]->at(i).IsInitialized() &&
          out->at(i).data() != x[0]->at(i).data()) {
        in_place = false;
        break;
      }
    }
  } else {
    in_place = false;
  }
  for (size_t i = in_place ? 1 : 0; i < x.size(); ++i) {
    auto* in_array = x.at(i);

    for (size_t j = 0; j < in_array->size(); ++j) {
      if (in_array->at(j).IsInitialized() && (in_array->at(j).numel() != 0)) {
        if (j >= out->size()) {
          out->resize(j + 1);
        }
        if (!out->at(j).IsInitialized() || (out->at(j).numel() == 0)) {
          Copy<Context>(dev_ctx,
                        in_array->at(j),
                        in_array->at(j).place(),
                        false,
                        &out->at(j));
          out->at(j).set_lod(in_array->at(j).lod());
        } else {
          PADDLE_ENFORCE_EQ(
              out->at(j).lod(),
              in_array->at(j).lod(),
              phi::errors::InvalidArgument(
                  "The lod message between inputs[%d] and"
                  " outputs[%d] must be same, but now is not same.",
                  j,
                  j));
          auto in = EigenVector<T>::Flatten(in_array->at(j));
          auto result = EigenVector<T>::Flatten(out->at(j));
          result.device(*dev_ctx.eigen_device()) = result + in;
        }
      }
    }
  }
}

}  // namespace phi
