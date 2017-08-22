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

#include "paddle/operators/functor/math_functor.h"
#include "paddle/framework/eigen.h"

namespace paddle {
namespace operators {
namespace functor {

template <typename T>
struct Set<platform::CPUPlace, T> {
  void operator()(const T alpha, framework::Tensor* Y,
                  platform::DeviceContext* context) {
    int N = product(Y->dims());
    T* YData = Y->mutable_data<T>(context->GetPlace());
    if (alpha == static_cast<T>(0)) {
      memset(YData, 0, N * sizeof(T));
    } else {
      framework::EigenVector<T, Eigen::RowMajor, Eigen::DenseIndex>::Flatten(*Y)
          .setConstant(alpha);
    }
  }
};

template struct Set<platform::CPUPlace, float>;
template struct Set<platform::CPUPlace, double>;

}  // namespace functor
}  // namespace operators
}  // namespace paddle
