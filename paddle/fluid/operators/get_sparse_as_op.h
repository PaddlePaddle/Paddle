/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"

namespace paddle {
namespace operators {

using DDim = framework::DDim;

constexpr int64_t kNoPadding = -1;

template <typename T>
class GetSparseAsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    using LoDTensor = framework::LoDTensor;
    using SelectedRows = framework::SelectedRows;

    auto *w_t = context.Input<LoDTensor>("W");
    auto *x_s = context.Input<SelectedRows>("X");
    auto *out_s = context.Output<SelectedRows>("Out");

    out_s->set_height(x_s->height());
    out_s->set_rows(x_s->rows());
    out_s->mutable_value()->Resize(x_s->value().dims());

    auto *output_t = out_s->mutable_value();
    auto *output = output_t->mutable_data<T>(context.GetPlace());

    int64_t row_number = w_t->dims()[0];
    int64_t row_width = w_t->dims()[1];
    auto *table = w_t->data<T>();

    int64_t ids_numel = x_s->rows().size();
    auto ids_v = x_s->rows();

    for (int64_t i = 0; i < ids_numel; ++i) {
      PADDLE_ENFORCE_LT(ids_v[i], row_number);
      PADDLE_ENFORCE_GE(ids_v[i], 0, "ids_v %d", i);
      memcpy(output + i * row_width, table + ids_v[i] * row_width,
             row_width * sizeof(T));
    }
  }
};

}  // namespace operators
}  // namespace paddle
