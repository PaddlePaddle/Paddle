/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

template <typename T>
static inline void random_permate(T* data_ptr, int num, unsigned int seed) {
  auto engine = framework::GetCPURandomEngine(seed);
  for (int i = 0; i < num; ++i) {
    data_ptr[i] = static_cast<T>(i);
  }

  std::shuffle(data_ptr, data_ptr + num, *engine);
}

template <typename DeviceContext, typename T>
class RandpermKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int n = ctx.Attr<int>("n");
    unsigned int seed = static_cast<unsigned int>(ctx.Attr<int>("seed"));
    framework::Variable* out_var = ctx.OutputVar("Out");
    framework::Tensor* out_tensor =
        framework::GetMutableLoDTensorOrSelectedRowsValueFromVar(out_var);

    if (platform::is_cpu_place(ctx.GetPlace())) {
      T* out_data = out_tensor->mutable_data<T>(platform::CPUPlace());
      random_permate<T>(out_data, n, seed);

    } else {
      framework::Tensor tmp_tensor;
      tmp_tensor.Resize(phi::make_ddim({n}));
      T* tmp_data = tmp_tensor.mutable_data<T>(platform::CPUPlace());
      random_permate<T>(tmp_data, n, seed);
      framework::TensorCopy(tmp_tensor, ctx.GetPlace(), out_tensor);
    }
  }
};

}  // namespace operators
}  // namespace paddle
