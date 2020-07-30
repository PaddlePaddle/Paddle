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
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/distributed/large_scale_kv.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using SelectedRows = framework::SelectedRows;

template <typename DeviceContext, typename T>
class LookupSparseTableGradSplitKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const SelectedRows* in_grad = context.Input<SelectedRows>("Grad");

    // merge duplicated rows if any.
    // The rows of grad_merge_ptr have been sorted inside MergeAdd functor
    framework::SelectedRows tmp_grad_merge;
    const framework::SelectedRows* grad_merge_ptr;
    math::scatter::MergeAdd<DeviceContext, T> merge_func;
    merge_func(context.template device_context<DeviceContext>(), *in_grad,
               &tmp_grad_merge, true);
    grad_merge_ptr = &tmp_grad_merge;

    std::vector<int64_t> in_rows;
    in_rows.reserve(grad_merge_ptr->rows().size());
    std::copy(grad_merge_ptr->rows().begin(), grad_merge_ptr->rows().end(),
              std::back_inserter(in_rows));

    auto* out_row = context.Output<Tensor>("Row");
    out_row->Resize(
        framework::make_ddim({static_cast<int64_t>(in_rows.size()), 1}));
    out_row->mutable_data<int64_t>(context.GetPlace());
    framework::TensorFromVector(in_rows, context.device_context(), out_row);

    auto in_value = grad_merge_ptr->value();
    std::vector<T> ins_vector;
    framework::TensorToVector(in_value, context.device_context(), &ins_vector);
    auto dims = in_value.dims();

    auto is_entry = context.Attr<bool>("is_entry");
    auto tablename = context.Attr<std::string>("tablename");

    if (is_entry) {
      auto* ins = distributed::LargeScaleKV::GetInstance();
      std::vector<int64_t> ids;
      ins->Get(tablename)->GetEntry(in_rows, &ids);

      for (auto& id : ids) {
        auto it = std::find(in_rows.begin(), in_rows.end(), id);
        if (it == in_rows.end()) {
          PADDLE_THROW(platform::errors::OutOfRange(
              "the input key should be exists. But received %d.", id));
        }

        auto distance =
            static_cast<int64_t>(std::distance(in_rows.begin(), it));
        std::fill(ins_vector.data() + distance * dims[1],
                  ins_vector.data() + dims[1], 0.0);
      }
    }

    auto* out_v = context.OutputVar("Value");
    out_v->Clear();
    auto* out_t = out_v->GetMutable<framework::LoDTensor>();
    out_t->mutable_data<T>(context.GetPlace());
    framework::TensorFromVector(ins_vector, context.device_context(), out_t);
    out_t->Resize(dims);
  }
};
}  // namespace operators
}  // namespace paddle
