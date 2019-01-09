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
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using SelectedRows = framework::SelectedRows;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
using platform::CPUDeviceContext;
using platform::CUDADeviceContext;

template <typename DeviceContext, typename T>
class ClipByNormFunctor;

template <typename T>
class ClipByNormFunctor<CPUDeviceContext, T> {
 public:
  explicit ClipByNormFunctor(const CPUDeviceContext& ctx) {}
  void operator()(const T* x, T* out, const size_t& numel, const T& max_norm) {
    T x_norm(0);
    for (size_t i = 0; i < numel; ++i) {
      x_norm += x[i] * x[i];
    }
    x_norm = std::sqrt(x_norm);
    T scaling(0);
    if (x_norm > max_norm) {
      scaling = max_norm / x_norm;
    } else {
      scaling = static_cast<T>(1);
    }
    for (size_t i = 0; i < numel; ++i) {
      out[i] = x[i] * scaling;
    }
  }
};

#ifdef PADDLE_WITH_CUDA
template <typename T>
class ClipByNormFunctor<CUDADeviceContext, T> {
 public:
  explicit ClipByNormFunctor(const CUDADeviceContext& ctx) : ctx_(ctx) {}
  void operator()(const T* x, T* out, const size_t& numel, const T& max_norm);

 private:
  const CUDADeviceContext& ctx_;
};
#endif

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

    auto& dev_ctx = context.template device_context<DeviceContext>();
    ClipByNormFunctor<DeviceContext, T> functor(dev_ctx);
    functor(input->data<T>(), output->mutable_data<T>(context.GetPlace()),
            input->numel(), max_norm);
  }
};

}  // namespace operators
}  // namespace paddle
