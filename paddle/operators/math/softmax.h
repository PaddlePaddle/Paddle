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
#include "paddle/framework/operator.h"
#include "paddle/framework/tensor.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class SoftmaxFunctor {
 public:
  void operator()(const framework::ExecutionContext& context,
                  const framework::Tensor* X, framework::Tensor* Y) {
    auto logits = EigenMatrix<T>::From(*X);
    auto softmax = EigenMatrix<T>::From(*Y);

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
}  // namespace math
}  // namespace operators
}  // namespace paddle
