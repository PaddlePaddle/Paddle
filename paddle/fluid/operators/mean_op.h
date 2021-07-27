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
#include "paddle/fluid/framework/pten_utils.h"

// only can include the headers in paddle/pten/api dirs
#include "paddle/pten/api/dev/core.h"
#include "paddle/pten/api/dev/math.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenScalar = framework::EigenScalar<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
class MeanKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* out = context.Output<Tensor>("Out");
    auto& dev_ctx = context.device_context<DeviceContext>();

    auto pt_x =
        framework::MakeTensorImpl<pt::DenseTensor>(*x, x->place(), x->type());
    auto pt_out =
        framework::MakeTensorImpl<pt::DenseTensor>(*out, x->place(), x->type());

    // call new kernel
    pt::Mean<T>(dev_ctx, *pt_x.get(), pt_out.get());

    // share pt_out data to out
    framework::ShareTensorImpl(pt_out.get(), out);
  }
};

template <typename DeviceContext, typename T>
class MeanGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto OG = context.Input<Tensor>(framework::GradVarName("Out"));
    PADDLE_ENFORCE_EQ(OG->numel(), 1UL,
                      platform::errors::InvalidArgument(
                          "Mean Gradient should be scalar. But received "
                          "Out@Grad's elements num is %d.",
                          OG->numel()));
    auto IG = context.Output<Tensor>(framework::GradVarName("X"));
    IG->mutable_data<T>(context.GetPlace());

    T ig_size = static_cast<T>(IG->numel());
    Eigen::DSizes<int, 1> bcast(static_cast<int>(ig_size));
    EigenVector<T>::Flatten(*IG).device(
        *context.template device_context<DeviceContext>().eigen_device()) =
        (EigenVector<T>::From(*OG) / ig_size).broadcast(bcast);
  }
};

}  // namespace operators
}  // namespace paddle
