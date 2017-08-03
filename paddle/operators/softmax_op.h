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

#include "paddle/framework/ddim.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/tensor.h"
#include "paddle/operators/type_alias.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class SoftmaxKernel : public OpKernel {
public:
  void Compute(const ExecutionContext& context) const override {
    auto input = context.Input<Tensor>("X");
    auto output = context.Output<Tensor>("Y");
    output->mutable_data<T>(context.GetPlace());

    auto logits = EigenMatrix<T>::From(*input);
    auto softmax = EigenMatrix<T>::From(*output);

    const int kBatchDim = 0;
    const int kClassDim = 1;

    const int batch_size = logits.dimension(kBatchDim);
    const int num_classes = logits.dimension(kClassDim);

    Eigen::DSizes<int, 1> along_class(kClassDim);
    Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
    Eigen::DSizes<int, 2> one_by_class(1, num_classes);

    auto shifted_logits = (logits -
                           logits.maximum(along_class)
                               .eval()
                               .reshape(batch_by_one)
                               .broadcast(one_by_class));

    softmax.device(context.GetEigenDevice<Place>()) = shifted_logits.exp();

    softmax.device(context.GetEigenDevice<Place>()) =
        (softmax *
         softmax.sum(along_class)
             .inverse()
             .eval()
             .reshape(batch_by_one)
             .broadcast(one_by_class));
  }
};

template <typename Place, typename T>
class SoftmaxGradKernel : public OpKernel {
public:
  void Compute(const ExecutionContext& context) const override {
    std::shared_ptr<Tensor> scale_ = std::make_shared<Tensor>();

    auto Y = context.Input<Tensor>("Y");
    auto dY = context.Input<Tensor>(framework::GradVarName("Y"));
    auto dX = context.Output<Tensor>(framework::GradVarName("X"));
    dX->mutable_data<T>(context.GetPlace());

    const int batch_size = Y->dims()[0];
    const int class_num = Y->dims()[1];

    Eigen::DSizes<int, 1> along_class(1);
    Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
    Eigen::DSizes<int, 2> one_by_class(1, class_num);

    auto Y_eigen = EigenMatrix<T>::From(*Y);
    auto dY_eigen = EigenMatrix<T>::From(*dY);
    auto dX_eigen = EigenMatrix<T>::From(*dX);
    auto place = context.GetEigenDevice<Place>();

    auto dot = (Y_eigen * dY_eigen)
                   .sum(along_class)
                   .eval()
                   .reshape(batch_by_one)
                   .broadcast(one_by_class);
    dX_eigen.device(place) = (dY_eigen - dot) * Y_eigen;
  }
};

}  // namespace operators
}  // namespace paddle
