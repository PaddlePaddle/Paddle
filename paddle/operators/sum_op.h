/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
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
#include "paddle/framework/eigen.h"
#include "paddle/framework/lod_tensor_array.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/math_function.h"
#include "paddle/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using SelectedRows = framework::SelectedRows;
using LoDTensor = framework::LoDTensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename Place, typename T>
class SumKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto in_vars = context.MultiInputVar("X");
    int N = in_vars.size();
    auto out_var = context.OutputVar("Out");

    bool in_place = out_var == in_vars[0];

    if (out_var->IsType<framework::LoDTensor>()) {
      auto *out = context.Output<Tensor>("Out");
      out->mutable_data<T>(context.GetPlace());

      auto result = EigenVector<T>::Flatten(*out);

      if (!in_place) {
        math::SetConstant<Place, T> constant_functor;
        constant_functor(context.device_context(), out, 0.0);
      }

      math::SelectedRowsAddToTensor<Place, T> functor;
      auto place = context.GetEigenDevice<Place>();
      // If in_place, just skip the first tensor
      for (int i = in_place ? 1 : 0; i < N; i++) {
        if (in_vars[i]->IsType<framework::LoDTensor>()) {
          auto &in_t = in_vars[i]->Get<framework::LoDTensor>();
          auto in = EigenVector<T>::Flatten(in_t);
          result.device(place) = result + in;
        } else if (in_vars[i]->IsType<framework::SelectedRows>()) {
          auto &in_t = in_vars[i]->Get<framework::SelectedRows>();
          functor(context.device_context(), in_t, out);
        } else {
          PADDLE_THROW("Variable type must be LoDTensor/SelectedRows.");
        }
      }
    } else if (out_var->IsType<framework::SelectedRows>()) {
      PADDLE_ENFORCE(!in_place, "SelectedRows not support inplace sum now");
      auto *out = context.Output<SelectedRows>("Out");
      auto *out_value = out->mutable_value();

      // Runtime InferShape
      size_t first_dim = 0;
      for (int i = 0; i < N; i++) {
        first_dim += in_vars[i]->Get<SelectedRows>().rows().size();
      }
      auto in_dim = in_vars[0]->Get<SelectedRows>().value().dims();
      auto in_dim_vec = framework::vectorize(in_dim);
      in_dim_vec[0] = static_cast<int64_t>(first_dim);

      out_value->Resize(framework::make_ddim(in_dim_vec));
      out_value->mutable_data<T>(context.GetPlace());

      math::SelectedRowsAddTo<Place, T> functor;

      int64_t offset = 0;
      for (int i = 0; i < N; i++) {
        PADDLE_ENFORCE_EQ(out->height(),
                          in_vars[i]->Get<SelectedRows>().height())
        functor(context.device_context(), in_vars[i]->Get<SelectedRows>(),
                offset, out);
        offset += in_vars[i]->Get<SelectedRows>().value().numel();
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
              out_array[i].CopyFrom(in_array[i], in_array[i].place(),
                                    context.device_context());
              out_array[i].set_lod(in_array[i].lod());
            } else {
              PADDLE_ENFORCE(out_array[i].lod() == in_array[i].lod());
              auto in = EigenVector<T>::Flatten(in_array[i]);
              auto result = EigenVector<T>::Flatten(out_array[i]);
              result.device(context.GetEigenDevice<Place>()) = result + in;
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
