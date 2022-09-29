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
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class PositiveNegativePairKernel : public framework::OpKernel<T> {
 public:
  struct PredictionResult {
    PredictionResult(T score, T label, T weight)
        : score(score), label(label), weight(weight) {}
    T score;
    T label;
    T weight;
  };

  void Compute(const framework::ExecutionContext& context) const override {
    auto score_t = context.Input<phi::DenseTensor>("Score");
    auto label_t = context.Input<phi::DenseTensor>("Label");
    auto query_t = context.Input<phi::DenseTensor>("QueryID");
    auto acc_positive_t =
        context.Input<phi::DenseTensor>("AccumulatePositivePair");
    auto acc_negative_t =
        context.Input<phi::DenseTensor>("AccumulateNegativePair");
    auto acc_neutral_t =
        context.Input<phi::DenseTensor>("AccumulateNeutralPair");
    auto positive_t = context.Output<phi::DenseTensor>("PositivePair");
    auto negative_t = context.Output<phi::DenseTensor>("NegativePair");
    auto neutral_t = context.Output<phi::DenseTensor>("NeutralPair");
    auto weight_t = context.Input<phi::DenseTensor>("Weight");

    auto score = score_t->data<T>();
    auto label = label_t->data<T>();
    auto query = query_t->data<int64_t>();
    const T* weight = nullptr;
    if (weight_t != nullptr) {
      weight = weight_t->data<T>();
    }
    T* positive = positive_t->mutable_data<T>(context.GetPlace());
    T* negative = negative_t->mutable_data<T>(context.GetPlace());
    T* neutral = neutral_t->mutable_data<T>(context.GetPlace());

    auto score_dim = score_t->dims();
    auto batch_size = score_dim[0];
    auto width = score_dim[1];
    auto column = context.Attr<int32_t>("column");
    if (column < 0) {
      column += width;
    }

    // construct document instances for each query: Query => List[<score#0,
    // label#0, weight#0>, ...]
    std::unordered_map<int64_t, std::vector<PredictionResult>> predictions;
    for (auto i = 0; i < batch_size; ++i) {
      if (predictions.find(query[i]) == predictions.end()) {
        predictions.emplace(
            std::make_pair(query[i], std::vector<PredictionResult>()));
      }
      predictions[query[i]].emplace_back(score[i * width + column],
                                         label[i],
                                         weight_t != nullptr ? weight[i] : 1.0);
    }

    // for each query, accumulate pair counts
    T pos = 0, neg = 0, neu = 0;
    if (acc_positive_t != nullptr && acc_negative_t != nullptr &&
        acc_neutral_t != nullptr) {
      pos = acc_positive_t->data<T>()[0];
      neg = acc_negative_t->data<T>()[0];
      neu = acc_neutral_t->data<T>()[0];
    }
    auto evaluate_one_list =
        [&pos, &neg, &neu](std::vector<PredictionResult> vec) {
          for (auto ite1 = vec.begin(); ite1 != vec.end(); ++ite1) {
            for (auto ite2 = ite1 + 1; ite2 != vec.end(); ++ite2) {
              if (ite1->label == ite2->label) {  // labels are equal, ignore.
                continue;
              }
              T w = (ite1->weight + ite2->weight) * 0.5;
              if (ite1->score == ite2->score) {
                neu += w;
              }
              (ite1->score - ite2->score) * (ite1->label - ite2->label) > 0.0
                  ? pos += w
                  : neg += w;
            }
          }
        };
    for (auto prediction : predictions) {
      evaluate_one_list(prediction.second);
    }
    *positive = pos;
    *negative = neg;
    *neutral = neu;
  }
};

}  // namespace operators
}  // namespace paddle
