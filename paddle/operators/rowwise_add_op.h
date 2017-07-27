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
#include "paddle/operators/type_alias.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class RowWiseAddKernel : public OpKernel {
public:
  void Compute(const KernelContext& context) const override {
    auto in0 = context.Input(0)->Get<Tensor>();
    auto in1 = context.Input(1)->Get<Tensor>();
    auto* out = context.Output(0)->GetMutable<Tensor>();
    out->mutable_data<T>(context.GetPlace());

    auto input = EigenMatrix<T>::From(in0);
    auto bias = EigenVector<T>::From(in1);
    auto output = EigenMatrix<T>::From(*out);

    const int bias_size = bias.dimension(0);
    const int rest_size = input.size() / bias_size;
    Eigen::DSizes<int, 1> one_d(input.size());
    Eigen::DSizes<int, 1> bcast(rest_size);
    output.reshape(one_d).device(*(context.GetEigenDevice<Place>())) =
        input.reshape(one_d) + bias.broadcast(bcast).reshape(one_d);
  }
};

}  // namespace operators
}  // namespace paddle
