/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <iostream>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/sampler.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/ddim.h"

namespace paddle {
namespace operators {
namespace math {

/* UNDERSTAND: utility function to adjust probability for unique sampling,
return whatever as it is if not using unique samping */
template <typename T>
static T adjust_prob(const T prob, const int num_samples, const int num_tries) {
  if (num_samples == num_tries) {
    return prob * num_samples;
  } else {
    return -expm1(num_tries * log1p(-prob));
  }
}

template <typename DeviceContext, typename T>
class SampleWithProb {
 public:
  void operator()(const DeviceContext& context,
                  const Sampler& sampler,
                  const std::size_t num_samples,
                  const phi::DenseTensor* L,
                  phi::DenseTensor* S,
                  phi::DenseTensor* P) {
    // UNDERSTAND: dimension issues
    const auto& lbl_dim = L->dims();
    const int batch_size = lbl_dim[0];
    const int num_true = lbl_dim[1];
    const int num_sampled_classes = num_true + num_samples;
    framework::DDim ret_dim{batch_size, num_sampled_classes};

    // UNDERSTAND: raw data view
    const int64_t* label_data = L->data<int64_t>();
    int64_t* samples_data =
        S->mutable_data<int64_t>(ret_dim, context.GetPlace());
    T* probabilities_data = P->mutable_data<T>(ret_dim, context.GetPlace());

    // temp sets for unique sampling
    std::unordered_set<int64_t> tmp_samples;
    int j = 0;  // column index
    // add true labels, not that efficient
    while (j < num_true) {
      for (int i = 0; i < batch_size; ++i) {
        auto samples_index = i * num_sampled_classes + j;
        auto v = label_data[i * num_true + j];
        samples_data[samples_index] = v;
        probabilities_data[samples_index] = sampler.Probability(v);
      }
      ++j;
    }

    // sample num_samles unique samples for an example, note that they are not
    // all negative samples
    tmp_samples.clear();
    int num_tries = 0;
    while (j < num_sampled_classes) {
      ++num_tries;
      auto v = sampler.Sample();
      auto insert_ok = tmp_samples.insert(v).second;
      if (!insert_ok) {
        continue;
      }
      auto p = sampler.Probability(v);
      for (int i = 0; i < batch_size; ++i) {
        auto samples_index = i * num_sampled_classes + j;
        samples_data[samples_index] = v;
        probabilities_data[samples_index] = p;
      }
      ++j;
    }

    // compute Q(y|x), because of unique sampling, probabilities need to be
    // adjusted
    for (int k = 0; k < num_sampled_classes; ++k) {
      for (int i = 0; i < batch_size; ++i) {
        auto samples_index = i * num_sampled_classes + k;
        probabilities_data[samples_index] = adjust_prob(
            probabilities_data[samples_index], num_samples, num_tries);
      }
    }
  }
};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename T>
class GPUSampleWithProb {
 public:
  void operator()(const phi::GPUContext& context,
                  const int seed,
                  const int dict_size,
                  const bool uniq,
                  const std::size_t num_samples,
                  const phi::DenseTensor* L,
                  phi::DenseTensor* S,
                  phi::DenseTensor* P);
};
#endif
}  // namespace math
}  // namespace operators
}  // namespace paddle
