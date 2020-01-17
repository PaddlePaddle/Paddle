/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <cmath>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/sampler.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using Sampler = math::Sampler;
using DDim = framework::DDim;
using LoD = framework::LoD;
using LoDAndOffset = std::pair<LoD, std::pair<size_t, size_t>>;

template <typename DeviceContext, typename T>
class TDMSamplerKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *input_var = context.InputVar("Input");
    auto *travel_var = context.InputVar("Travel");
    auto *layer_var = context.InputVar("Layer");

    auto neg_samples_num_vec =
        context.Attr<std::vector<int64_t>>("neg_samples_num_list");
    auto output_positive_flag = context.Attr<bool>("output_positive");

    // get all tensor
    auto &input_tensor = input_var->Get<framework::LoDTensor>();
    auto &travel_lod_tensor = travel_var->Get<framework::LoDTensor>();
    auto &layer_lod_tensor = layer_var->Get<framework::LoDTensor>();
    auto *out_tensor = context.Output<framework::LoDTensor>("Out");
    auto *label_tensor = context.Output<framework::LoDTensor>("Labels");

    // get dimension
    int64_t input_ids_num = input_tensor.numel();
    auto travel_lod = travel_lod_tensor.lod();
    auto layer_lod = layer_lod_tensor.lod();
    auto layer_nums = neg_samples_num_vec.size();

    int64_t sample_res_length = 0;
    for (int64_t layer_idx = 0; layer_idx < layer_nums; ++layer_idx) {
      sample_res_length +=
          (neg_samples_num_vec[layer_idx] + (int64_t)output_positive_flag);
    }

    auto layer_node_offset = layer_lod[0];

    // get all data
    int64_t *input_data = const_cast<int64_t *>(input_tensor.data<int64_t>());
    int64_t *travel_data =
        const_cast<int64_t *>(travel_lod_tensor.data<int64_t>());
    int64_t *layer_data =
        const_cast<int64_t *>(layer_lod_tensor.data<int64_t>());
    int64_t *output_data = out_tensor->data<int64_t>();
    int64_t *label_data = label_tensor->data<int64_t>();

    // generate uniform sampler

    auto seed = context.Attr<int>("seed");

    for (int64_t i = 0; i < input_ids_num; ++i) {
      // find leaf node travel path
      auto lod_and_offset = framework::GetSubLoDAndAbsoluteOffset(
          travel_lod, input_data[i], input_data[i] + 1, 0);
      std::pair<size_t, size_t> pos_offset = lod_and_offset.second;
      size_t sampling_depth =
          (pos_offset.second - pos_offset.first) > layer_nums
              ? layer_nums
              : (pos_offset.second - pos_offset.first);

      // nce sample, layer by layer

      int64_t offset = 0;
      for (size_t layer_idx = 0; layer_idx < sampling_depth; ++layer_idx) {
        int64_t sample_num = neg_samples_num_vec[layer_idx];
        int64_t node_nums =
            layer_node_offset[layer_idx + 1] - layer_node_offset[layer_idx];
        Sampler *sampler = new math::UniformSampler(node_nums, seed);
        // If output positive, add itself
        if (output_positive_flag) {
          output_data[i * sample_res_length + offset] =
              travel_data[pos_offset.first + layer_idx];
          label_data[i * sample_res_length + offset] = 1;
          offset += 1;
        }

        // Sampling at layer, until samples enough
        for (int64_t sample_index = 0; sample_index < sample_num;
             ++sample_index) {
          // Avoid sampling to positive samples
          int64_t sample_res = 0;
          do {
            sample_res = sampler->Sample();
          } while (travel_data[pos_offset.first + layer_idx] ==
                   layer_data[layer_node_offset[layer_idx] + sample_res]);

          output_data[i * sample_res_length + offset] =
              layer_data[layer_node_offset[layer_idx] + sample_res];
          label_data[i * sample_res_length + offset] = 0;
          offset += 1;
        }  // end layer nce
        delete sampler;
      }  // end one input nce

      while (sampling_depth < layer_nums) {
        // padding extra neg example
        int64_t sample_num =
            neg_samples_num_vec[sampling_depth] + (int64_t)output_positive_flag;
        int64_t node_nums = layer_node_offset[sampling_depth + 1] -
                            layer_node_offset[sampling_depth];
        Sampler *sampler = new math::UniformSampler(node_nums, seed);
        for (int64_t sample_index = 0; sample_index < sample_num;
             ++sample_index) {
          int64_t sample_res = sampler->Sample();
          output_data[i * sample_res_length + offset] =
              layer_data[layer_node_offset[sampling_depth] + sample_res];
          label_data[i * sample_res_length + offset] = 0;
          offset += 1;
        }
        delete sampler;
        sampling_depth += 1;
      }
    }  // end all input nce
  }
};

}  // namespace operators
}  // namespace paddle
