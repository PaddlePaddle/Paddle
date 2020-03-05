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

#include <gflags/gflags.h>
#include <cmath>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DDim = framework::DDim;
using LoD = framework::LoD;

template <typename DeviceContext, typename T>
class FindByIndexKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *input_var = ctx.InputVar("Input");
    auto *index_var = ctx.InputVar("Index");

    auto &input_tensor = input_var->Get<LoDTensor>();
    auto &index_tensor = index_var->Get<LoDTensor>();
    auto input_dims = input_tensor.dims();
    auto index_dims = index_tensor.dims();

    int batch_size = input_dims[0];
    auto value_length = input_dims[1];
    auto index_length = index_dims[1];

    // int input_ids_num = input_tensor.numel();
    int index_ids_num = index_tensor.numel();

    std::vector<T> res{};

    auto *input_data = input_tensor.data<T>();
    auto *index_data = index_tensor.data<int64_t>();
    for (int i = 0; i < index_ids_num; i++) {
      int b = floor(i / index_length);
      int v_i = b * value_length + static_cast<int>(index_data[i]);
      T v = input_data[v_i];
      VLOG(1) << "Find by index: b= " << b << " v_i= " << v_i << " v= " << v;
      res.push_back(v);
    }

    auto *out_var = ctx.OutputVar("Out");
    auto *out_tensor = out_var->GetMutable<framework::LoDTensor>();
    auto ddim = framework::make_ddim({batch_size, index_length});
    out_tensor->Resize(ddim);
    auto *out_data = out_tensor->mutable_data<T>(ctx.GetPlace());

    memcpy(out_data, &res[0], sizeof(T) * index_ids_num);
  }
};

}  // namespace operators
}  // namespace paddle
