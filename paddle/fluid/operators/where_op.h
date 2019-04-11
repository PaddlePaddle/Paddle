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
#include <functional>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T>
class CPUWhereKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* condition = context.Input<framework::Tensor>("Condition");
    auto* out = context.Output<framework::Tensor>("Out");

    const bool* cond_data = condition->data<bool>();
    int64_t numel = condition->numel();
    auto dims = condition->dims();
    const int64_t kRank = dims.size();

    // calculate true_num
    size_t true_num = std::accumulate(cond_data, cond_data + numel, 0LL);

    // setup out shape
    if (true_num != 0) {
      out->Resize(
          framework::make_ddim({static_cast<int64_t>(true_num), kRank}));
    } else {
      out->Resize(framework::make_ddim({}));
      return;
    }
    out->mutable_data<T>(context.GetPlace());

    // fill in output with coordinate of true element
    auto out_data = EigenMatrix<int64_t>::From(*out);

    int stride[kRank];
    stride[kRank - 1] = 1;
    for (int i = kRank - 2; i >= 0; i--) {
      stride[i] = stride[i + 1] * dims[i + 1];
    }

    true_num = 0;
    for (int64_t i = 0; i < numel; i++) {
      if (cond_data[i]) {
        int64_t index = i;
        for (int j = 0; j < kRank; j++) {
          out_data(true_num, j) = index / stride[j];
          index -= out_data(true_num, j) * stride[j];
        }
        true_num += 1;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
