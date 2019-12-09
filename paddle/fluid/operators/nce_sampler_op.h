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
    int sampler_type = context.Attr<int>("sampler");
    int seed = context.Attr<int>("seed");
    int num_total_classes = context.Attr<int>("num_total_classes");

    Sampler *sampler;
    switch (sampler_type) {
      case 0: {
        sampler = new math::UniformSampler(num_total_classes - 1, seed);
        break;
      }
      case 1: {
        sampler = new math::LogUniformSampler(num_total_classes - 1, seed);
        break;
      }
      case 2: {
        auto dist_probs = context.Input<Tensor>("CustomDistProbs");
        auto dist_alias = context.Input<Tensor>("CustomDistAlias");
        auto dist_alias_probs = context.Input<Tensor>("CustomDistAliasProbs");

        PADDLE_ENFORCE_EQ(
            dist_probs->numel(), num_total_classes,
            "ShapeError: The number of elements in Input(CustomDistProbs) "
            "should be equal to the number of total classes. But Received: "
            "Input(CustomDistProbs).numel() = %d, Attr(num_total_classes) "
            "= %d.",
            dist_probs->numel(), num_total_classes);
        PADDLE_ENFORCE_EQ(
            dist_alias->numel(), num_total_classes,
            "ShapeError: The number of elements in Input(CustomDistAlias) "
            "should be equal to the number of total classes. But Received: "
            "Input(CustomDistAlias).numel() = %d, Attr(num_total_classes) "
            "= %d.",
            dist_alias->numel(), num_total_classes);
        PADDLE_ENFORCE_EQ(
            dist_alias_probs->numel(), num_total_classes,
            "ShapeError: The number of elements in Input(CustomDistAliasProbs) "
            "should be equal to the number of total classes. But Received: "
            "Input(CustomDistAliasProbs).numel() = %d, "
            "Attr(num_total_classes) = %d.",
            dist_alias_probs->numel(), num_total_classes);

        const float *probs_data = dist_probs->data<float>();
        const int *alias_data = dist_alias->data<int>();
        const float *alias_probs_data = dist_alias_probs->data<float>();

        sampler = new math::CustomSampler(num_total_classes - 1, probs_data,
                                          alias_data, alias_probs_data, seed);
      }
      default: { PADDLE_THROW("Unsupported SamplerType."); }
    }
    auto output = context.Output<Tensor>("Out");
    auto sample_labels_dims = output->dims();
    int64_t *sample_labels_data =
        output->mutable_data<int64_t>(context.GetPlace());

    int64_t index = 0;
    for (int64_t i = 0; i < sample_labels_dims[0]; ++i) {
      for (int64_t j = 0; j < sample_labels_dims[1]; j++) {
        sample_labels_data[index++] = sampler->Sample();
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle
