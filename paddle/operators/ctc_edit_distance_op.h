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

template <typename Place, typename T>
class CTCEditDistanceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* out_t = ctx.Output<framework::Tensor>("Out");

    auto* x1_t = ctx.Input<framework::Tensor>("X1");
    auto* x2_t = ctx.Input<framework::Tensor>("X2");

    out_t->mutable_data<float>(ctx.GetPlace());

    auto normalized = ctx.Attr<bool>("normalized");

    auto m = x1_t->numel();
    auto n = x2_t->numel();
    float distance = 0.0;
    if (m == 0) {
      distance = n;
    } else if (n == 0) {
      distance = m;
    } else {
      framework::Tensor dist_t;
      dist_t.Resize({m + 1, n + 1});
      dist_t.mutable_data<T>(ctx.GetPlace());
      auto dist = dist_t.data<T>();
      auto x1 = x1_t->data<T>();
      auto x2 = x2_t->data<T>();
      for (size_t i = 0; i < m + 1; ++i) {
        dist[i * (n + 1)] = i;
      }
      for (size_t j = 0; j < n + 1; ++j) {
        dist[j] = j;
      }
      for (size_t i = 1; i < m + 1; ++i) {
        for (size_t j = 1; j < n + 1; ++j) {
          int cost = x1[i - 1] == x2[j - 1] ? 0 : 1;
          int dels = dist[(i - 1) * (n + 1) + j] + 1;
          int ins = dist[i * (n + 1) + (j - 1)] + 1;
          int subs = dist[(i - 1) * (n + 1) + (j - 1)] + cost;
          dist[i * (n + 1) + j] = std::min(dels, std::min(ins, subs));
        }
      }
      distance = dist[m * (n + 1) + n];
    }

    if (normalized) {
      distance = distance / n;
    }
    auto out = out_t->data<float>();
    out[0] = distance;
  }
};

}  // namespace operators
}  // namespace paddle
