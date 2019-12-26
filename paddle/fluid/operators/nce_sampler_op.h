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
      auto factor = context.Attr<std::float>("factor");
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

      Sampler *sampler;
      sampler = new math::CustomSampler(num_total_classes - 1, probs_data,
                                        alias_data, alias_probs_data, seed);

      auto *output_var = context.OutputVar("Out");
      auto *output = output_var->GetMutable<framework::LoDTensor>();
      auto sample_labels_dims = output->dims();
      int64_t *sample_labels_data =
          output->mutable_data<int64_t>(context.GetPlace());

      int64_t index = 0;
      for (int64_t i = 0; i < sample_labels_dims[0]; ++i) {
        for (int64_t j = 0; j < sample_labels_dims[1]; j++) {
          sample_labels_data[index++] = sampler->Sample();
        }
      }
      delete sampler;
    }
  }
};
}  // namespace operators
}  // namespace paddle
