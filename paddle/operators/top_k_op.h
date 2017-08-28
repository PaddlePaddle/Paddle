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
#include <algorithm>
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenScalar = framework::EigenScalar<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename Place, typename T>
class TopkKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // Get the top k elements of each row of input tensor
    // FIXME: only deal with matrix(2d tensor).
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");

    output->mutable_data<T>(ctx.GetPlace());

    auto X = EigenMatrix<T>::From(*input);
    auto Out = EigenMatrix<T>::From(*output);
    // k is determined by Attr
    const int beam = static_cast<T>(context.op_.GetAttr<AttrType>("k"));

    const int height = X.dimension(0);
    const int width = X.dimension(1);
    T* s = Out.device(ctx.GetEigenDevice<Place>()).data();

    for (size_t i = 0; i < height; i++) {
      std::vector<std::pair<real, size_t>> vec;
      for (size_t j = 0; j < width; j++) {
        vec.push_back(std::pair<real, size_t>(a[i * width + j], j));
      }

      std::partial_sort(
          vec.begin(), vec.begin() + beam, vec.end(),
          [](const std::pair<real, size_t>& l,
             const std::pair<real, size_t>& r) { return l.first > r.first; });
      for (size_t j = 0; j < beam; j++) {
        // t[i * beam + j] = vec[j].first;
        s[i * beam + j] = vec[j].second;
      }
    }
  }
};

template <typename Place, typename T>
class MeanGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto OG = ctx.Input<Tensor>(framework::GradVarName("Out"));
    PADDLE_ENFORCE(framework::product(OG->dims()) == 1,
                   "Mean Gradient should be scalar");
    auto IG = ctx.Output<Tensor>(framework::GradVarName("X"));
    IG->mutable_data<T>(ctx.GetPlace());

    T ig_size = (T)framework::product(IG->dims());
    Eigen::DSizes<int, 1> bcast(ig_size);

    EigenVector<T>::Flatten(*IG).device(ctx.GetEigenDevice<Place>()) =
        (EigenVector<T>::From(*OG) / ig_size).broadcast(bcast);
  }
};

}  // namespace operators
}  // namespace paddle
