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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/math/algorithm.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using SelectedRows = framework::SelectedRows;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenScalar = framework::EigenScalar<T, MajorType, IndexType>;

template <typename T>
struct ClipByNormFunctor {
  ClipByNormFunctor(const T* scale, const T* x, T* out, const size_t& numel)
      : scale_(scale), x_(x), out_(out), numel_(numel) {}
  HOSTDEVICE inline void operator()(int64_t idx) const {
    out_[numel_ - 1 - idx] = scale_[0] * x_[numel_ - 1 - idx];
  }
  const T* scale_;
  const T* x_;
  T* out_;
  const size_t numel_;
};

template <typename DeviceContext, typename T>
class ClipByNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto max_norm = context.Attr<T>("max_norm");
    auto in_var = context.InputVar("X");

    Tensor* output = nullptr;
    const Tensor* input = nullptr;
    if (in_var->IsType<framework::LoDTensor>()) {
      input = context.Input<Tensor>("X");

      output = context.Output<Tensor>("Out");
      output->mutable_data<T>(context.GetPlace());
    } else if (in_var->IsType<SelectedRows>()) {
      auto* x = context.Input<SelectedRows>("X");

      // merge ids in selected rows first
      math::scatter::MergeAdd<DeviceContext, T> merge_func;
      SelectedRows* merged_input =
          const_cast<framework::Scope&>(context.scope())
              .Var()
              ->GetMutable<SelectedRows>();
      merge_func(context.template device_context<DeviceContext>(), *x,
                 merged_input);
      input = &(merged_input->value());

      SelectedRows* output_selected_rows = context.Output<SelectedRows>("Out");
      output_selected_rows->set_rows(merged_input->rows());
      output_selected_rows->set_height(merged_input->height());
      output = output_selected_rows->mutable_value();
      output->Resize(merged_input->value().dims());
      output->mutable_data<T>(context.GetPlace());
    } else {
      PADDLE_THROW("Unexpected branch, input variable type is %s",
                   framework::ToTypeName(in_var->Type()));
    }

    PADDLE_ENFORCE_NOT_NULL(input);

    auto x = EigenVector<T>::Flatten(detail::Ref(input));
    auto out_dims = output->dims();
    output->Resize(framework::make_ddim({1}));
    auto norm = EigenScalar<T>::From(detail::Ref(output));
    auto x_norm = x.square().sum().sqrt();
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();

    auto temp = (x_norm <= max_norm).template cast<T>();
    norm.device(place) = temp + (static_cast<T>(1) - temp) * max_norm / x_norm;

    output->Resize(out_dims);
    ClipByNormFunctor<T> functor(output->data<T>(), input->data<T>(),
                                 output->mutable_data<T>(context.GetPlace()),
                                 input->numel());
    platform::ForRange<DeviceContext> for_range(
        static_cast<const DeviceContext&>(context.device_context()),
        input->numel());
    for_range(functor);
  }
};

}  // namespace operators
}  // namespace paddle
