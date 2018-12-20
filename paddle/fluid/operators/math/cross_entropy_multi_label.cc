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

#include "paddle/fluid/operators/math/cross_entropy_multi_label.h"

namespace paddle {
namespace operators {
namespace math {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

// UNDERSTAND：template specification of CPUDeviceContext
template <typename T>
class CrossEntropyMultiLabelFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& ctx, framework::Tensor* out,
                  const framework::Tensor* prob,
                  const framework::Tensor* labels, const int ignore_index) {
    const int batch_size = prob->dims()[0];
    const int class_num = prob->dims()[1];
    const int true_num = labels->dims()[1];

    // UNDERSTAND: in order to compute them raw, unwrap them from Tensor
    const T* prob_data = prob->data<T>();
    const int64_t* label_data = labels->data<int64_t>();
    T* loss_data = out->data<T>();

    for (int i = 0; i < batch_size; ++i) {
      /* UNDERSTAND: at this circumstance, label is int Tensor (B, NT) and
      like softlabel case, ignore_index does not function
      */
      loss_data[i] = 0;
      for (int j = 0; j < true_num; ++j) {
        int lbl = label_data[i * true_num + j];
        PADDLE_ENFORCE_GE(lbl, 0);
        PADDLE_ENFORCE_LT(lbl, class_num);
        int index = i * class_num + lbl;
        loss_data[i] -=
            math::TolerableValue<T>()(std::log(prob_data[index]) / true_num);
      }
      loss_data[i] = math::TolerableValue<T>()(loss_data[i]);
    }
  }
};

// UNDERSTAND: template instantiation
template class CrossEntropyMultiLabelFunctor<platform::CPUDeviceContext, float>;
template class CrossEntropyMultiLabelFunctor<platform::CPUDeviceContext,
                                             double>;
// UNDERSTAND: where is the GradFunctor? in op, well, let it be
}  // namespace math
}  // namespace operators
}  // namespace paddle
