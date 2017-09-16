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
#include "paddle/platform/transform.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
using platform::Transform;

template <typename T>
class Prelu_functor {
 public:
  explicit Prelu_functor(const T& alpha) : alpha_(alpha) {}

  HOSTDEVICE T operator()(const T& X) const {
    if (X > 0)
      return X;
    else
      return X * alpha_;
  }

 private:
  T alpha_;
};

template <typename Place, typename T, typename AttrType = T>
class PReluKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<Tensor>("X");
    auto* Out = context.Output<Tensor>("Out");

    const T* X_ptr = X->data<T>();
    T* O_ptr = Out->mutable_data<T>(context.GetPlace());

    auto alpha = static_cast<T>(context.Attr<AttrType>("alpha"));

    int numel = X->numel();

    auto place = context.GetPlace();
    Transform(place, X_ptr, X_ptr + numel, O_ptr, Prelu_functor<T>(alpha));
  }
};

template <typename T>
class Prelu_Grad_functor {
 public:
  explicit Prelu_Grad_functor(const T& alpha) : alpha_(alpha) {}

  HOSTDEVICE T operator()(const T& Out, const T& dOut) const {
    if (Out > 0)
      return dOut;
    else
      return dOut * alpha_;
  }

 private:
  T alpha_;
};

template <typename Place, typename T, typename AttrType = T>
class PReluGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* dX = context.Output<Tensor>(framework::GradVarName("X"));
    auto* dO = context.Input<Tensor>(framework::GradVarName("Out"));

    auto* Out = context.Input<Tensor>("Out");

    auto alpha = static_cast<T>(context.Attr<AttrType>("alpha"));

    T* dX_ptr = dX->mutable_data<T>(context.GetPlace());
    const T* dO_ptr = dO->data<T>();
    const T* O_ptr = Out->data<T>();
    int numel = dX->numel();

    auto place = context.GetPlace();
    Transform(place, O_ptr, O_ptr + numel, dO_ptr, dX_ptr,
              Prelu_Grad_functor<T>(alpha));
  }
};

}  // namespace operators
}  // namespace paddle
