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
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenScalar = framework::EigenScalar<T, MajorType, IndexType>;

template <typename Place, typename T>
class AccuracyKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* inference = ctx.Input<Tensor>("Inference");
    auto* label = ctx.Input<Tensor>("Label");
    auto* accuracy = ctx.Output<Tensor>("Accuracy");

    float* accuracy_data = accuracy->mutable_data<float>(ctx.GetPlace());

    const T* inference_data = inference->data<T>();
    const T* label_data = label->data<T>();

    size_t num_samples = inference->dims()[0];
    size_t class_dim = inference->dims()[1];
    *accuracy_data = 0.0f;

    if (num_samples == 0) {
      return;
    }

    int num_correct = 0;
    // assume inference is already the topk of the output
    for (size_t i = 0; i < num_samples; ++i) {
      PADDLE_ENFORCE_GE(label_data[i], 0, "label must >= 0");
      for (size_t j = 0; j < class_dim; ++j) {
        if (inference_data[i * class_dim + j] == label_data[i]) {
          ++num_correct;
          break;
        }
      }
    }

    // FIXME(typhoonzero): we don't accumulate the accuracy for now.
    *accuracy_data =
        static_cast<float>(num_correct) / static_cast<float>(num_samples);
  }
};

}  // namespace operators
}  // namespace paddle
