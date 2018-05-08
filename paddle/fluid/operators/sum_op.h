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
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using SelectedRows = framework::SelectedRows;
using LoDTensor = framework::LoDTensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
class SumKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto in_vars = context.MultiInputVar("X");
    int N = in_vars.size();
    auto out_var = context.OutputVar("Out");

    bool in_place = out_var == in_vars[0];

    if (out_var->IsType<framework::LoDTensor>()) {
      auto *out = context.Output<LoDTensor>("Out");
      if (!in_place) {
        out->mutable_data<T>(context.GetPlace());
      }
      auto result = EigenVector<T>::Flatten(*out);
      if (!in_place) {
        math::SetConstant<DeviceContext, T> constant_functor;
        constant_functor(context.template device_context<DeviceContext>(), out,
                         0.0);
      }

      math::SelectedRowsAddToTensor<DeviceContext, T> functor;
      auto &place =
          *context.template device_context<DeviceContext>().eigen_device();
      // If in_place, just skip the first tensor
      for (int i = in_place ? 1 : 0; i < N; i++) {
        if (in_vars[i]->IsType<framework::LoDTensor>()) {
          auto &in_t = in_vars[i]->Get<framework::LoDTensor>();
          if (in_t.numel() == 0) {
            continue;
          }
          auto in = EigenVector<T>::Flatten(in_t);
          result.device(place) = result + in;
        } else if (in_vars[i]->IsType<framework::SelectedRows>()) {
          auto &in_t = in_vars[i]->Get<framework::SelectedRows>();
          functor(context.template device_context<DeviceContext>(), in_t, out);
        } else {
          PADDLE_THROW("Variable type must be LoDTensor/SelectedRows.");
        }
      }
    } else if (out_var->IsType<framework::SelectedRows>()) {
      std::unique_ptr<framework::SelectedRows> in0;
      if (in_place) {
        // If is in_place, we store the input[0] to in0
        auto &in_sel0 = in_vars[0]->Get<SelectedRows>();
        auto &rows = in_sel0.rows();
#ifdef PADDLE_WITH_CUDA
        std::vector<int64_t> rows_in_cpu;
        rows_in_cpu.reserve(rows.size());
        for (auto item : rows) {
          rows_in_cpu.push_back(item);
        }
        in0.reset(new framework::SelectedRows(rows_in_cpu, in_sel0.height()));
#else
        in0.reset(new framework::SelectedRows(rows, in_sel0.height()));
#endif
        in0->mutable_value()->ShareDataWith(in_sel0.value());
      }

      auto get_selected_row = [&](size_t i) -> const SelectedRows & {
        if (i == 0 && in0) {
          return *in0.get();
        } else {
          return in_vars[i]->Get<SelectedRows>();
        }
      };

      auto *out = context.Output<SelectedRows>("Out");
      out->mutable_rows()->clear();
      auto *out_value = out->mutable_value();

      // Runtime InferShape
      size_t first_dim = 0;
      for (int i = 0; i < N; i++) {
        auto &sel_row = get_selected_row(i);
        first_dim += sel_row.rows().size();
      }
      auto in_dim =
          framework::vectorize(get_selected_row(N - 1).value().dims());
      in_dim[0] = static_cast<int64_t>(first_dim);

      out_value->Resize(framework::make_ddim(in_dim));

      // if all the input sparse vars are empty, no need to
      // merge these vars.
      if (first_dim == 0UL) {
        return;
      }
      out_value->mutable_data<T>(context.GetPlace());

      math::SelectedRowsAddTo<DeviceContext, T> functor;

      int64_t offset = 0;
      for (int i = 0; i < N; i++) {
        auto &sel_row = get_selected_row(i);
        if (sel_row.rows().size() == 0) {
          continue;
        }
        PADDLE_ENFORCE_EQ(out->height(), sel_row.height());
        functor(context.template device_context<DeviceContext>(), sel_row,
                offset, out);
        offset += sel_row.value().numel();
      }
    } else if (out_var->IsType<framework::LoDTensorArray>()) {
      auto &out_array = *out_var->GetMutable<framework::LoDTensorArray>();
      for (size_t i = in_place ? 1 : 0; i < in_vars.size(); ++i) {
        PADDLE_ENFORCE(in_vars[i]->IsType<framework::LoDTensorArray>(),
                       "Only support all inputs are TensorArray");
        auto &in_array = in_vars[i]->Get<framework::LoDTensorArray>();

        for (size_t i = 0; i < in_array.size(); ++i) {
          if (in_array[i].numel() != 0) {
            if (i >= out_array.size()) {
              out_array.resize(i + 1);
            }
            if (out_array[i].numel() == 0) {
              framework::TensorCopy(in_array[i], in_array[i].place(),
                                    context.device_context(), &out_array[i]);
              out_array[i].set_lod(in_array[i].lod());
            } else {
              PADDLE_ENFORCE(out_array[i].lod() == in_array[i].lod());
              auto in = EigenVector<T>::Flatten(in_array[i]);
              auto result = EigenVector<T>::Flatten(out_array[i]);
              result.device(*context.template device_context<DeviceContext>()
                                 .eigen_device()) = result + in;
            }
          }
        }
      }
    } else {
      PADDLE_THROW("Unexpected branch, output variable type is %s",
                   out_var->Type().name());
    }
  }
};
}  // namespace operators
}  // namespace paddle
