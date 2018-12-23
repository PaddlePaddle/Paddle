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
#include <iostream>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/sampler.h"

namespace paddle {
namespace operators {
namespace math {

using Tensor = framework::Tensor;

/* UNDERSTAND: utility functor to adjust probability for unique sampling, return
whatever as it is if not using unique samping */
template <typename T>
static T adjust_prob(const bool unique, const T prob, const int num_samples,
                     const int num_tries) {
  if (!unique) {
    return prob * num_samples;
  } else {
    return -expm1(num_tries * log1p(-prob));
  }
}

template <typename DeviceContext, typename T>
class SampleWithProb {
 public:
  void operator()(const DeviceContext& context, const Sampler& sampler,
                  const bool unique, const std::size_t num_samples,
                  const bool remove_accidental_hits, const Tensor* L, Tensor* S,
                  Tensor* P, std::vector<int64_t>* hits,
                  std::vector<int64_t>* num_tries_vec,
                  const std::vector<int64_t>& avoid_indices) {
    // UNDERSTAND: dimension issues
    const auto lbl_dim = L->dims();
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
    std::unordered_set<int64_t> tmp_true_labels;
    std::unordered_set<int64_t> tmp_samples;

    // build avoid_set first
    std::unordered_set<int64_t> avoid_set;
    avoid_set.insert(avoid_indices.begin(), avoid_indices.end());

    int num_tries = 0;

    for (int i = 0; i < batch_size; ++i) {
      // init again
      tmp_samples.clear();
      tmp_true_labels.clear();
      num_tries = 0;

      // add true labels to true_label set
      tmp_true_labels.insert(label_data + i * num_true,
                             label_data + (i + 1) * num_true);

      int j = 0;  // column index
      // add true labels, not that efficient
      while (j < num_true) {
        auto samples_index = i * num_sampled_classes + j;
        auto v = label_data[i * num_true + j];
        samples_data[samples_index] = v;
        probabilities_data[samples_index] = sampler.Probability(v);
        ++j;
      }

      // add (possibly not)negative labels to samples
      // may be use a more efficient sampler to sample N unique samples.
      if (unique) {
        while (j < num_sampled_classes) {
          ++num_tries;
          auto v = sampler.Sample();
          auto samples_index = i * num_sampled_classes + j;
          if (avoid_set.find(v) == avoid_set.end()) {
            if (tmp_samples.insert(v).second) {
              samples_data[samples_index] = v;
              probabilities_data[samples_index] = sampler.Probability(v);
              if (remove_accidental_hits &&
                  tmp_true_labels.find(v) != tmp_true_labels.end()) {
                hits->push_back(samples_index);
              }
              ++j;
            }
          }
        }
      } else {
        PADDLE_ENFORCE_EQ(
            avoid_indices.size(), 0,
            "Avoid indices is only supported when using sampling.");
        while (j < num_sampled_classes) {
          auto v = sampler.Sample();
          if (avoid_set.find(v) == avoid_set.end()) {
            auto samples_index = i * num_sampled_classes + j;
            samples_data[samples_index] = v;
            probabilities_data[samples_index] = sampler.Probability(v);
            if (remove_accidental_hits &&
                tmp_true_labels.find(v) != tmp_true_labels.end()) {
              hits->push_back(samples_index);
            }
            ++j;
          }
        }
        num_tries = num_samples;
      }
      num_tries_vec->push_back(num_tries);

      // compute Q(y|x), so called probabilities here
      for (int k = 0; k < num_sampled_classes; ++k) {
        auto samples_index = i * num_sampled_classes + k;
        probabilities_data[samples_index] = adjust_prob(
            unique, probabilities_data[samples_index], num_samples, num_tries);
      }
    }
  }
};
}  // namespace math
}  // namespace operators
}  // namespace paddle
