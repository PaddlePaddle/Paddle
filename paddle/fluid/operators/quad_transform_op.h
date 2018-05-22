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

#pragma once
#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class QuadTransformCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    auto* in = ctx.Input<Tensor>("Input");
    auto in_dims = in->dims();
    const T* in_data = in->data<T>();
    auto* out = ctx.Output<Tensor>("Output");
    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    int batch_size = in_dims[0];
    int height = in_dims[2];
    int width = in_dims[3];
    int id = 0;
    for (int id_n = 0; id_n < batch_size * 8; ++id_n) {
      for (int id_h = 0; id_h < height; ++id_h) {
        for (int id_w = 0; id_w < width; ++id_w) {
          id = id_n * height * width + width * id_h + id_w;
          if (id_n % 2 == 0) {
            out_data[id] = in_data[id] + id_w;
          } else {
            out_data[id] = in_data[id] + id_h;
          }
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
