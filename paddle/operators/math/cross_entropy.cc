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

#include "paddle/operators/math/cross_entropy.h"

namespace paddle {
namespace operators {
namespace math {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T>
class CrossEntropyFunctor<platform::CPUPlace, T> {
 public:
  void operator()(const platform::DeviceContext& ctx, framework::Tensor* out,
                  const framework::Tensor* prob,
                  const framework::Tensor* labels, const bool softLabel) {
    const int batch_size = prob->dims()[0];
    if (softLabel) {
      auto in = EigenMatrix<T>::From(*prob);
      auto lbl = EigenMatrix<T>::From(*labels);
      auto loss = EigenMatrix<T>::From(*out);

      loss.device(*ctx.GetEigenDevice<platform::CPUPlace>()) =
          -((lbl * in.log().unaryExpr(math::TolerableValue<T>()))
                .sum(Eigen::DSizes<int, 1>(1))
                .reshape(Eigen::DSizes<int, 2>(batch_size, 1)));
    } else {
      const int class_num = prob->dims()[1];
      const T* prob_data = prob->data<T>();
      T* loss_data = out->data<T>();

      const int* label_data = labels->data<int>();
      for (int i = 0; i < batch_size; ++i) {
        int index = i * class_num + label_data[i];
        loss_data[i] = -math::TolerableValue<T>()(std::log(prob_data[index]));
      }
    }
  }
};

template class CrossEntropyFunctor<platform::CPUPlace, float>;
}  // namespace math
}  // namespace operators
}  // namespace paddle
