/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;

template <typename T>
class CEmbeddingOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *table_t = ctx.Input<LoDTensor>("W");
    auto *ids_t = ctx.Input<LoDTensor>("Ids");
    auto *output_t = ctx.Output<LoDTensor>("Out");
    const int64_t start_idx = ctx.Attr<int64_t>("start_index");

    VLOG(10) << "table_dims:" << table_t->dims();

    PADDLE_ENFORCE_EQ(table_t->dims().size(), 2,
                      platform::errors::InvalidArgument(
                          "npu only accept the dims of table_t == 2"));

    std::vector<T> ids_t_vec;
    framework::TensorToVector(*ids_t, &ids_t_vec);

    std::vector<T> table_t_vec;
    framework::TensorToVector(*table_t, &table_t_vec);

    const size_t height = table_t->dims()[0];
    const size_t width = table_t->dims()[1];
    std::vector<std::vector<T>> out;
    out.resize(ids_t->numel());

    for (size_t i = 0; i < ids_t_vec.size(); i++) {
      size_t id = ids_t_vec[i];
      size_t local = id - start_idx;

      std::vector<T> tmp(width, static_cast<T>(0.0));
      if (local >= 0 && local < height) {
        for (size_t w = 0; w < width; w++) {
          tmp[w] = table_t_vec[local * height + w];
        }
      }
      out[i] = std::move(tmp);
    }

    auto dims = output_t->dims();
    framework::TensorFromVector(out, output_t);
    output_t->Resize(dims);
  }
};

template <typename T>
class CEmbeddingGradOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const int64_t start_idx = context.Attr<int64_t>("start_index");
    auto ids_t = context.Input<LoDTensor>("Ids");
    auto d_output_t = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto table_t = context.Output<LoDTensor>(framework::GradVarName("W"));

    VLOG(10) << "table_dims:" << table_t->dims();

    std::vector<T> ids_t_vec;
    framework::TensorToVector(*ids_t, &ids_t_vec);

    std::vector<T> table_t_vec;
    framework::TensorToVector(*table_t, &table_t_vec);

    std::vector<T> d_output_vec;
    framework::TensorToVector(*d_output_t, &d_output_vec);

    const size_t height = table_t->dims()[0];
    const size_t width = table_t->dims()[1];

    for (size_t i = 0; i < ids_t_vec.size(); i++) {
      size_t id = ids_t_vec[i];
      size_t local = id - start_idx;

      if (local >= 0 && local < height) {
        for (size_t w = 0; w < width; w++) {
          table_t_vec[local * height + w] = d_output_vec[i * height + w];
        }
      }
    }

    auto dims = table_t->dims();
    framework::TensorFromVector(table_t_vec, table_t);
    table_t->Resize(dims);
  }
};

}  // namespace operators
}  // namespace paddle
