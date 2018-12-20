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
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/sampler.h"

namespace paddle {
namespace operators {
namespace math {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class SampleWithProb {
 public:
  void operator()(const DeviceContext& context, const Sampler& sampler,
                  const std::size_t num_classes, const std::size_t num_samples,
                  const Tensor* L, Tensor* S, Tensor* P,
                  const std::vector<int>& custom_negative_classes) {
    // UNDERSTAND: dimension issues
    const int kBatchDim = 0;
    const int kClassDim = 1;
    auto lbl_dim = L->dims();
    const int batch_size = lbl_dim[kBatchDim];
    const int num_true = lbl_dim[kClassDim];
    const int num_sampled_classes = num_true + num_samples;
    framework::DDim ret_dim{batch_size, num_sampled_classes};

    // UNDERSTAND: raw data view
    const int64_t* label_data = L->data<int64_t>();
    int64_t* samples_data =
        S->mutable_data<int64_t>(ret_dim, context.GetPlace());
    T* probabilities_data = P->mutable_data<T>(ret_dim, context.GetPlace());

    std::unordered_set<int64_t> tmp_true_labels;
    std::unordered_set<int64_t> tmp_samples;

    // build custom_negative_classes_set
    std::unordered_set<int64_t> custom_negative_classes_set;
    for (const auto& v : custom_negative_classes) {
      custom_negative_classes_set.insert(v);
    }

    for (int i = 0; i < batch_size; ++i) {
      tmp_samples.clear();
      tmp_true_labels.clear();
      // add true labels to true_label set
      for (int j = 0; j < num_true; ++j) {
        tmp_true_labels.insert(label_data[i * num_true + j]);
      }

      // add true labels
      for (int k = 0; k < num_true; ++k) {
        auto samples_index = i * num_sampled_classes + k;
        auto v = label_data[i * num_true + k];
        samples_data[samples_index] = v;
        probabilities_data[samples_index] = sampler.Probability(v);
      }

      /* add custom negative sample, used for unittest, but custom
      negative sampls should be really negative */
      for (std::size_t j = 0; j < custom_negative_classes.size(); ++j) {
        auto samples_index = i * num_sampled_classes + num_true + j;
        const auto& v = custom_negative_classes[j];
        samples_data[samples_index] = v;
        probabilities_data[samples_index] = sampler.Probability(v);
      }

      // add (possibly) negative labels to samples
      // TODO(chenfeiyu) may be use a more efficient sampler to sample N unique
      // samples.
      for (int k = num_true + custom_negative_classes.size();
           k < num_sampled_classes; ++k) {
        auto v = sampler.Sample();
        if ((tmp_true_labels.find(v) == tmp_true_labels.end()) &&
            (custom_negative_classes_set.find(v) ==
             custom_negative_classes_set.end())) {
          if (tmp_samples.insert(v).second) {
            auto samples_index = i * num_sampled_classes + k;
            samples_data[samples_index] = v;
            probabilities_data[samples_index] = sampler.Probability(v);
          }
        }
      }
    }
  }
};
}  // namespace math
}  // namespace operators
}  // namespace paddle
