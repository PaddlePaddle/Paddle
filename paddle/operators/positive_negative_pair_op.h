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
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename Place, typename T>
class PositiveNegativePairKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto score_t = context.Input<Tensor>("Score");
    auto label_t = context.Input<Tensor>("Label");
    auto query_t = context.Input<Tensor>("QueryId");
    auto positive_t = context.Output<Tensor>("PositivePair");
    auto negative_t = context.Output<Tensor>("NegativePair");
    auto neutral_t = context.Output<Tensor>("NeutralPair");

    auto score = score_t->data<float>();
    auto label = label_t->data<float>();
    auto query = query_t->data<int32_t>();

    T* positive = positive_t->mutable_data<T>(context.GetPlace());
    T* negative = negative_t->mutable_data<T>(context.GetPlace());
    T* neutral = neutral_t->mutable_data<T>(context.GetPlace());

    auto score_dim = score_t->dims();
    PADDLE_ENFORCE_GE(score_dim.size(), 1L,
                      "Rank of Score must be at least 1.");
    PADDLE_ENFORCE_LE(score_dim.size(), 2L,
                      "Rank of Score must be less or equal to 2.");
    auto batch_size = score_dim[0];
    auto width = score_dim.size() > 1 ? score_dim[1] : 1;

    // construct document instances for each query: Query => List[<score#0,
    // label#0>, ...]
    std::unordered_map<int, std::vector<std::pair<float, float>>> predictions;
    for (auto i = 0; i < batch_size; ++i) {
      if (predictions.find(query[i]) == predictions.end()) {
        predictions.emplace(
            std::make_pair(query[i], std::vector<std::pair<float, float>>()));
      }
      predictions[query[i]].push_back(
          std::make_pair(score[i * width + width - 1], label[i]));
    }

    // for each query, accumulate pair counts
    T pos = 0, neg = 0, neu = 0;
    auto evaluate_one_list = [&pos, &neg,
                              &neu](std::vector<std::pair<float, float>> vec) {
      for (auto ite1 = vec.begin(); ite1 != vec.end(); ++ite1) {
        for (auto ite2 = ite1 + 1; ite2 != vec.end(); ++ite2) {
          if (ite1->second == ite2->second) {  // labels are equal, ignore.
            continue;
          }
          if (ite1->first == ite2->first) {
            ++neu;
          }
          (ite1->first - ite2->first) * (ite1->second - ite2->second) > 0.0
              ? pos++
              : neg++;
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
