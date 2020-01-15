// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cmath>
#include <fstream>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/sampler.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using Sampler = math::Sampler;
using DDim = framework::DDim;

template <typename DeviceContext, typename T>
class NCESamplerKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    bool init_flag = context.Attr<bool>("init_flag");
    if (init_flag) {
      VLOG(1) << "start to init nce sampler";
      auto *dist_probs_var = context.OutputVar("CustomDistProbsInit");
      auto *dist_alias_var = context.OutputVar("CustomDistAliasInit");
      auto *dist_alias_probs_var =
          context.OutputVar("CustomDistAliasProbsInit");

      auto *dist_probs_init =
          dist_probs_var->GetMutable<framework::LoDTensor>();
      auto *dist_alias_init =
          dist_alias_var->GetMutable<framework::LoDTensor>();
      auto *dist_alias_probs_init =
          dist_alias_probs_var->GetMutable<framework::LoDTensor>();

      auto filename = context.Attr<std::string>("filename");
      auto factor = context.Attr<float>("factor");

      std::ifstream fin(filename);
      int64_t f_count, f_count_pow, total_count;
      total_count = 0;
      std::vector<int64_t> _word_count;
      while (fin >> f_count) {
        f_count_pow = static_cast<int64_t>(pow(f_count, factor));
        _word_count.push_back(f_count_pow);
        total_count += f_count_pow;
      }
      int k = _word_count.size();

      float *_prob = dist_probs_init->mutable_data<float>(context.GetPlace());
      int *_j = dist_alias_init->mutable_data<int>(context.GetPlace());
      float *_q =
          dist_alias_probs_init->mutable_data<float>(context.GetPlace());

      for (int i = 0; i < k; ++i) {
        _prob[i] = static_cast<float>(_word_count[i]) / total_count;
      }

      std::set<int> S;
      std::set<int>::iterator s_it;
      std::set<int> L;
      std::set<int>::iterator l_it;
      for (int i = 0; i < k; ++i) {
        _q[i] = k * _prob[i];
        _j[i] = -1;
        if (_q[i] < 1.0) {
          S.insert(i);
        } else {
          L.insert(i);
        }
      }

      while (S.size() > 0 && L.size() > 0) {
        s_it = S.begin();
        int s = *s_it;
        l_it = L.begin();
        int l = *l_it;
        _j[s] = l;
        S.erase(s_it);
        _q[l] = _q[l] + _q[s] - 1.0;
        if (_q[l] < 1.0) {
          S.insert(l);
          L.erase(l_it);
        }
      }

      for (s_it = S.begin(); s_it != S.end(); ++s_it) {
        _q[*s_it] = 1.0;
      }

      for (l_it = L.begin(); l_it != L.end(); ++l_it) {
        _q[*l_it] = 1.0;
      }
      VLOG(1) << "NCE Sampler Op finish initialization. Dict Size is " << k
              << "; and Total Word Count is " << total_count;
    } else {
      auto *dist_probs_var = context.InputVar("CustomDistProbs");
      auto *dist_alias_var = context.InputVar("CustomDistAlias");
      auto *dist_alias_probs_var = context.InputVar("CustomDistAliasProbs");

      auto &dist_probs = dist_probs_var->Get<framework::LoDTensor>();
      auto &dist_alias = dist_alias_var->Get<framework::LoDTensor>();
      auto &dist_alias_probs =
          dist_alias_probs_var->Get<framework::LoDTensor>();

      int seed = context.Attr<int>("seed");
      int num_total_classes = context.Attr<int>("num_total_classes");
      int num_neg_samples = context.Attr<int>("num_neg_samples");

      PADDLE_ENFORCE_EQ(
          dist_probs.numel(), num_total_classes,
          "ShapeError: The number of elements in Input(CustomDistProbs) "
          "should be equal to the number of total classes. But Received: "
          "Input(CustomDistProbs).numel() = %d, Attr(num_total_classes) "
          "= %d.",
          dist_probs.numel(), num_total_classes);
      PADDLE_ENFORCE_EQ(
          dist_alias.numel(), num_total_classes,
          "ShapeError: The number of elements in Input(CustomDistAlias) "
          "should be equal to the number of total classes. But Received: "
          "Input(CustomDistAlias).numel() = %d, Attr(num_total_classes) "
          "= %d.",
          dist_alias.numel(), num_total_classes);
      PADDLE_ENFORCE_EQ(
          dist_alias_probs.numel(), num_total_classes,
          "ShapeError: The number of elements in Input(CustomDistAliasProbs) "
          "should be equal to the number of total classes. But Received: "
          "Input(CustomDistAliasProbs).numel() = %d, "
          "Attr(num_total_classes) = %d.",
          dist_alias_probs.numel(), num_total_classes);

      const float *probs_data = dist_probs.data<float>();
      const int *alias_data = dist_alias.data<int>();
      const float *alias_probs_data = dist_alias_probs.data<float>();

      std::unordered_set<int64_t> pos_samples;
      if (context.HasInput("PositiveSamples")) {
        auto *pos_var = context.InputVar("PositiveSamples");
        auto &pos_tensor = pos_var->Get<framework::LoDTensor>();
        const int64_t *pos_samples_data = pos_tensor.data<int64_t>();
        auto total_nums = pos_tensor.numel();
        for (auto i = 0; i < total_nums; i++) {
          pos_samples.insert(pos_samples_data[i]);
        }
      }

      Sampler *sampler;
      sampler = new math::CustomSampler(num_total_classes - 1, probs_data,
                                        alias_data, alias_probs_data, seed);

      auto *output_var = context.OutputVar("Out");
      auto *output = output_var->GetMutable<framework::LoDTensor>();
      int64_t *sample_labels_data =
          output->mutable_data<int64_t>(context.GetPlace());

      for (int64_t index = 0; index < num_neg_samples; ++index) {
        auto res = sampler->Sample();
        while (pos_samples.find(res) != pos_samples.end()) {
          res = sampler->Sample();
        }
        sample_labels_data[index] = res;
      }
      delete sampler;
    }
  }
};
}  // namespace operators
}  // namespace paddle
