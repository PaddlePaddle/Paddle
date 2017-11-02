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
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/cross_entropy.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T>
class CrossEntropyOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "This kernel only runs on CPU.");
    const Tensor* x = ctx.Input<Tensor>("X");
    const Tensor* labels = ctx.Input<Tensor>("Label");
    Tensor* y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());

    math::CrossEntropyFunctor<platform::CPUPlace, T>()(
        ctx.device_context(), y, x, labels, ctx.Attr<bool>("soft_label"));
  }
};

template <typename T>
class CrossEntropyGradientOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "This kernel only runs on CPU.");
    const Tensor* x = ctx.Input<Tensor>("X");
    const Tensor* dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const Tensor* label = ctx.Input<Tensor>("Label");
    Tensor* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    T* dx_data = dx->mutable_data<T>(ctx.GetPlace());

    int64_t class_num = x->dims()[1];
    if (ctx.Attr<bool>("soft_label")) {
      auto x_mat = EigenMatrix<T>::From(*x);
      auto dy_mat = EigenMatrix<T>::From(*dy);
      auto lbl_mat = EigenMatrix<T>::From(*label);
      auto dx_mat = EigenMatrix<T>::From(*dx);

      dx_mat.device(ctx.GetEigenDevice<platform::CPUPlace>()) =
          -(lbl_mat *
            dy_mat.broadcast(Eigen::DSizes<int64_t, 2>(1, class_num)) / x_mat);
    } else {
      int64_t batch_size = x->dims()[0];
      const T* dy_data = dy->data<T>();
      const T* x_data = x->data<T>();
      const int64_t* label_data = label->data<int64_t>();

      math::SetConstant<platform::CPUPlace, T> functor;
      functor(ctx.device_context(), dx, 0);

      for (int64_t i = 0; i < batch_size; ++i) {
        PADDLE_ASSERT(label_data[i] >= 0 || label_data[i] < class_num);
        int64_t index = i * class_num + label_data[i];
        dx_data[index] = -dy_data[i] / x_data[index];
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
