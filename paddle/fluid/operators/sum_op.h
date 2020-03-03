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
void SelectedRowsCompute(const framework::ExecutionContext &context) {
  auto in_vars = context.MultiInputVar("X");
  auto out_var = context.OutputVar("Out");
  bool in_place = out_var == in_vars[0];

  if (in_place && in_vars.size() < 2) {
    return;
  }

  std::vector<const paddle::framework::SelectedRows *> inputs;
  SelectedRows temp_in0;

  if (in_place) {
    auto &in0 = in_vars[0]->Get<SelectedRows>();
    temp_in0.set_height(in0.height());
    temp_in0.set_rows(in0.rows());
    framework::TensorCopy(in0.value(), in0.place(), context.device_context(),
                          temp_in0.mutable_value());
    inputs.push_back(&temp_in0);
    for (size_t i = 1; i < in_vars.size(); ++i) {
      auto &in = in_vars[i]->Get<SelectedRows>();
      if (in.rows().size() > 0) {
        inputs.push_back(&in);
      }
    }
  } else {
    for (auto &in_var : in_vars) {
      auto &in = in_var->Get<SelectedRows>();
      if (in.rows().size() > 0) {
        inputs.push_back(&in_var->Get<SelectedRows>());
      }
    }
  }

  auto *out = context.Output<SelectedRows>("Out");
  out->mutable_rows()->clear();

  bool has_data = false;
  for (auto &in : inputs) {
    if (in->rows().size() > 0) {
      has_data = true;
      break;
    }
  }
  if (has_data) {
    math::scatter::MergeAdd<DeviceContext, T> merge_add;
    merge_add(context.template device_context<DeviceContext>(), inputs, out);

    out->SyncIndex();

  } else {
    // no data, just set a empty out tensor.
    out->mutable_value()->mutable_data<T>(framework::make_ddim({0}),
                                          context.GetPlace());
  }
}

template <typename DeviceContext, typename T>
void LodTensorArrayCompute(const framework::ExecutionContext &context) {
  auto in_vars = context.MultiInputVar("X");
  auto out_var = context.OutputVar("Out");
  bool in_place = out_var == in_vars[0];
  auto &out_array = *out_var->GetMutable<framework::LoDTensorArray>();
  for (size_t i = in_place ? 1 : 0; i < in_vars.size(); ++i) {
    PADDLE_ENFORCE_EQ(in_vars[i]->IsType<framework::LoDTensorArray>(), true,
                      "Only support all inputs are TensorArray");
    auto &in_array = in_vars[i]->Get<framework::LoDTensorArray>();

    for (size_t i = 0; i < in_array.size(); ++i) {
      if (in_array[i].IsInitialized() && (in_array[i].numel() != 0)) {
        if (i >= out_array.size()) {
          out_array.resize(i + 1);
        }
        if (!out_array[i].IsInitialized() || (out_array[i].numel() == 0)) {
          framework::TensorCopy(in_array[i], in_array[i].place(),
                                context.device_context(), &out_array[i]);
          out_array[i].set_lod(in_array[i].lod());
        } else {
          PADDLE_ENFORCE_EQ(out_array[i].lod(), in_array[i].lod());
          auto in = EigenVector<T>::Flatten(in_array[i]);
          auto result = EigenVector<T>::Flatten(out_array[i]);
          result.device(*context.template device_context<DeviceContext>()
                             .eigen_device()) = result + in;
        }
      }
    }
  }
}

template <typename DeviceContext, typename T>
class SumKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto in_vars = context.MultiInputVar("X");
    size_t in_num = in_vars.size();
    auto out_var = context.OutputVar("Out");

    size_t item_size = 0;
    if (out_var->IsType<framework::LoDTensor>()) {
      auto *out = out_var->GetMutable<framework::LoDTensor>();
      auto *out_ptr = out->mutable_data<T>(context.GetPlace());

      std::vector<const T *> tensor;
      std::vector<size_t> select_row;
      for (size_t i = 0; i < in_num; i++) {
        if (in_vars[i]->IsType<framework::LoDTensor>()) {
          const auto &in_tensor = in_vars[i]->Get<framework::LoDTensor>();
          if (in_tensor.numel()) {
            const T *in_data = in_tensor.data<T>();
            tensor.push_back(in_data);
            item_size = in_tensor.numel();
          }
        } else if (in_vars[i]->IsType<framework::SelectedRows>()) {
          select_row.push_back(i);
        } else {
          PADDLE_THROW("Variable type must be LoDTensor/SelectedRows.");
        }
      }
      if (tensor.size()) {
        for (size_t i = 0; i < item_size; ++i) {
          T result_temp = tensor[0][i];
          for (size_t j = 1; j < tensor.size(); ++j) {
            auto *in0_ptr = tensor[j];
            result_temp = result_temp + in0_ptr[i];
          }
          out_ptr[i] = result_temp;
        }
      } else {
        math::SetConstant<DeviceContext, T> constant_functor;
        constant_functor(context.template device_context<DeviceContext>(), out,
                         static_cast<T>(0));
      }

      math::SelectedRowsAddToTensor<DeviceContext, T> functor;
      for (size_t i = 0; i < select_row.size(); i++) {
        auto &in_t = in_vars[select_row[i]]->Get<framework::SelectedRows>();
        functor(context.template device_context<DeviceContext>(), in_t, out);
      }
    } else if (out_var->IsType<framework::SelectedRows>()) {
      SelectedRowsCompute<DeviceContext, T>(context);
    } else if (out_var->IsType<framework::LoDTensorArray>()) {
      LodTensorArrayCompute<DeviceContext, T>(context);
    } else {
      PADDLE_THROW("Unexpected branch, output variable type is %s",
                   framework::ToTypeName(out_var->Type()));
    }
  }
};
}  // namespace operators
}  // namespace paddle
