// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_XPU
#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "paddle/fluid/operators/reduce_ops/reduce_op.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
void XPUReduce(const framework::ExecutionContext& context,
               std::function<int(xpu::Context*,
                                 const T*,
                                 T*,
                                 const std::vector<int>&,
                                 const std::vector<int>&)> func) {
  PADDLE_ENFORCE_EQ(
      platform::is_xpu_place(context.GetPlace()),
      true,
      platform::errors::Unavailable("This kernel only runs on XPU."));
  bool reduce_all = context.Attr<bool>("reduce_all");
  auto dims = context.Attr<std::vector<int>>("dim");
  auto* x = context.Input<phi::DenseTensor>("X");
  auto* y = context.Output<phi::DenseTensor>("Out");
  y->mutable_data<T>(context.GetPlace());
  auto& dev_ctx = context.template device_context<DeviceContext>();

  int out_dtype = context.Attr<int>("out_dtype");
  PADDLE_ENFORCE_EQ(out_dtype == -1,
                    true,
                    platform::errors::InvalidArgument(
                        "XPU only support out_dtype == -1 in reduce op."));

  const auto* x_data = x->data<T>();
  auto* y_data = y->data<T>();
  const auto& input_dim_size = x->dims().size();
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
    xdims[i] = x->dims()[i];
  }
  if (reduce_all) {
    for (int i = 0; i < input_dim_size; ++i) {
      reduce_dims.push_back(i);
    }
  } else {
    std::set<int> dims_set(true_dims.begin(), true_dims.end());
    for (auto i = 0; i < input_dim_size; i++) {
      if (dims_set.find(i) != dims_set.end()) {
        if (x->dims()[i] != 1) {
          reduce_dims.push_back(i);
        }
      }
    }
  }

  if (reduce_dims.size() == 0) {
    int r = xpu::copy<T>(
        dev_ctx.x_context(), x_data, y_data, x->numel() * sizeof(T));
    PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS,
                      true,
                      platform::errors::External("XPU copy in reduce op return "
                                                 "wrong value[%d %s].",
                                                 r,
                                                 XPUAPIErrorMsg[r]));
  } else {
    int r = func(dev_ctx.x_context(), x_data, y_data, xdims, reduce_dims);
    PADDLE_ENFORCE_EQ(
        r == xpu::Error_t::SUCCESS,
        true,
        platform::errors::External(
            "XPU reduce op return wrong value[%d %s].", r, XPUAPIErrorMsg[r]));
  }
}

}  // namespace operators
}  // namespace paddle
#endif
