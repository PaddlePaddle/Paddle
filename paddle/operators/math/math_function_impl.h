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

#include "paddle/framework/data_type.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {
namespace math {

template <typename Place, typename T>
void SetConstant<Place, T>::operator()(const platform::DeviceContext& context,
                                       framework::Tensor* tensor, T num) {
  auto t = framework::EigenVector<T>::Flatten(*tensor);
  t.device(*context.GetEigenDevice<platform::CPUPlace>()) =
      t.constant(static_cast<T>(num));
}

template <typename Place, typename T, int Rank>
void Transpose<Place, T, Rank>::operator()(
    const platform::DeviceContext& context, const framework::Tensor& in,
    framework::Tensor* out, const std::vector<int>& axis) {
  Eigen::array<int, Rank> permute;
  for (int i = 0; i < Rank; i++) {
    permute[i] = axis[i];
  }
  auto in_dim = in.dims();
  auto out_dim = out->dims();

  auto eigen_in = framework::EigenTensor<T, Rank>::From(in);
  auto eigen_out = framework::EigenTensor<T, Rank>::From(*out);
  auto* dev = context.GetEigenDevice<Place>();
  eigen_out.device(*dev) = eigen_in.shuffle(permute);
}
}
}
}
