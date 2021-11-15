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

// only can include the headers in paddle/top/api dirs
#include "paddle/pten/api/lib/utils/tensor_utils.h"
#include "paddle/pten/include/core.h"
#include "paddle/pten/include/math.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenScalar = framework::EigenScalar<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

/** [ Why still keep the original kernel implementation? ]
 *
 * Removal of the original kernel implementation and kernel registration needs
 * to ensure that the new kernel mechanism adapts to multiple sets of execution
 * mechanisms, including:
 *
 * 1. Executor and ParallelExecutor
 * 2. Dygraph OpBase (Tracer and Engine)
 * 3. New Executor
 * 4. Predictor
 * 5. NPU and XPU lack kernel and need to reuse CPU Kernel
 *
 * Removal of the original Kernel requires a more complete solution to ensure
 * that it will not affect the current execution system.
 * Currently, only the first two cases are adapted.
 *
 * The principle here is that the implementation in the kernel must reuse the
 * corresponding functions in the Tensor Operation library and cannot maintain
 * two copies of the code.
 */
template <typename DeviceContext, typename T>
class MeanKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* out = context.Output<Tensor>("Out");
    auto& dev_ctx = context.device_context<DeviceContext>();
    out->mutable_data<T>(x->place());

    auto pt_x = paddle::experimental::MakePtenDenseTensor(*x);
    auto pt_out = paddle::experimental::MakePtenDenseTensor(*out);

    // call new kernel
    pten::Mean<T>(dev_ctx, *pt_x.get(), pt_out.get());
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
