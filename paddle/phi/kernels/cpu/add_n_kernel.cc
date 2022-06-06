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

#include "paddle/phi/kernels/add_n_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void AddNKernel(const Context& dev_ctx,
                const std::vector<const DenseTensor*>& x,
                DenseTensor* out) {
  size_t in_num = x.size();
  bool in_place = out == x[0];
  auto* out_ptr = dev_ctx.template Alloc<T>(out);
  if (in_num >= 1 && x[0]->initialized()) {
    if (x[0]->numel() > 0) {
      in_place = (x[0]->data<T>() == out_ptr);
    }
  }

  auto result = EigenVector<T>::Flatten(*out);
  auto& place = *dev_ctx.eigen_device();
  int start = in_place ? 1 : 0;
  if (!in_place) {
    if ((in_num >= 2) && x[0]->initialized() && x[1]->initialized()) {
      auto& in_0 = *x[0];
      auto& in_1 = *x[1];
      if (in_0.numel() && in_1.numel()) {
        auto in_0_e = EigenVector<T>::Flatten(in_0);
        auto in_1_e = EigenVector<T>::Flatten(in_1);
        result.device(place) = in_0_e + in_1_e;
        start = 2;
      }
    }
    if (start != 2) {
      VLOG(10) << "Fill with constant = 0 in sum kernel.";
      funcs::SetConstant<Context, T> constant_functor;
      constant_functor(dev_ctx, out, static_cast<T>(0));
    }
  }

  // If in_place, just skip the first tensor
  for (size_t i = start; i < in_num; i++) {
    auto& in_t = *x[i];
    if (!in_t.initialized() || in_t.numel() == 0) {
      continue;
    }
    auto in = EigenVector<T>::Flatten(in_t);
    result.device(place) = result + in;
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(add_n,
                   CPU,
                   ALL_LAYOUT,
                   phi::AddNKernel,
                   float,
                   double,
                   int,
                   phi::dtype::bfloat16,
                   int64_t) {}
