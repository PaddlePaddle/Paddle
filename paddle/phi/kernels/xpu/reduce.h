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
#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace phi {

template <typename Context, typename T>
int XPUReduce(const Context& dev_ctx,
              const DenseTensor& x,
              const std::vector<int64_t>& dims,
              bool keep_dim,
              bool reduce_all,
              DenseTensor* out,
              std::function<int(xpu::Context*,
                                const T*,
                                T*,
                                const std::vector<int>&,
                                const std::vector<int>&)> func) {
  dev_ctx.template Alloc<T>(out);

  const auto* x_data = x.data<T>();
  auto* y_data = out->data<T>();
  const auto& input_dim_size = x.dims().size();
  std::vector<int> true_dims;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] < 0) {
      true_dims.push_back(dims[i] + input_dim_size);
    } else {
      true_dims.push_back(dims[i]);
    }
  }

  std::vector<int> reduce_dims;
  std::vector<int> xdims((input_dim_size));
  for (int i = 0; i < input_dim_size; ++i) {
    xdims[i] = x.dims()[i];
  }
  if (reduce_all) {
    for (int i = 0; i < input_dim_size; ++i) {
      reduce_dims.push_back(i);
    }
  } else {
    std::set<int> dims_set(true_dims.begin(), true_dims.end());
    for (auto i = 0; i < input_dim_size; i++) {
      if (dims_set.find(i) != dims_set.end()) {
        if (x.dims()[i] != 1) {
          reduce_dims.push_back(i);
        }
      }
    }
  }

  int r = xpu::SUCCESS;
  if (reduce_dims.size() == 0) {
    r = xpu::copy<T>(
        dev_ctx.x_context(), x_data, y_data, x.numel() * sizeof(T));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
  } else {
    r = func(dev_ctx.x_context(), x_data, y_data, xdims, reduce_dims);
  }
  return r;
}

}  // namespace phi
