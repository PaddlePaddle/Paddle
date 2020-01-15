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
    auto &input_tensor = input_var->Get<framework::Tensor>();
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
    std::vector<int64_t> layer_node_offset = layer_lod[0];

    // get all data
    int64_t *input_data = const_cast<int64_t *>(input_tensor.data<int64_t>());
    int64_t *travel_data =
        const_cast<int64_t *>(travel_lod_tensor.data<int64_t>);
    int64_t *layer_data = const_cast<int64_t *>(layer_lod_tensor.data<int64_t>);
    int64_t *output_data = out_tensor->data<int64_t>();
    int64_t *lable_data = lable_tensor->data<int64_t>();

    // generate uniform sampler
    auto node_nums = layer_lod_tensor.numel();
    auto seed = context.Attr<int>("seed");
    Sampler *sampler = new math::UniformSampler(node_nums - 1, seed);

    for (int64_t i = 0; i < input_ids_num; ++i) {
      // find leaf node travel path
      std::pair<size_t, size_t> pos_offset =
          find_leaf_travel(input_data[i], travel_lod);
      size_t sampling_depth =
          (pos_offset.second - pos_offset.first) > layer_nums
              ? layer_nums
              : (pos_offset.second - pos_offset.first);

      // nce sample, layer by layer
      int64_t offset = 0;
      for (size_t layer_idx = 0; layer_idx < sampling_depth; ++layer_idx) {
        int64_t sample_num = neg_samples_num_vec[layer_idx];

        // If output positive, add itself
        if (output_positive_flag) {
          output_data[i * sample_res_length + offset] =
              travel_data[pos_offset.first + layer_idx];
          lable_data[i * sample_res_length + offset] = 1;
          offset += 1;
        }

        // Sampling at layer, until samples enough
        for (int64_t sample_index = 0; sample_index < sample_num;
             ++sample_index) {
          // Avoid sampling to positive samples
          int64_t sample_res = 0;
          do {
            sample_res = sampler->Sample() % neg_samples_num_vec[layer_idx];
          } while (travel_data[pos_offset.first + layer_idx] ==
                   layer_data[layer_node_offset[layer_idx] + sample_res]);

          output_data[i * sample_res_length + offset] =
              layer_data[layer_node_offset[layer_idx] + sample_res];
          lable_data[i * sample_res_length + offset] = 0;
          offset += 1;
        }  // end layer nce
      }    // end one input nce

      while (sampling_depth < layer_nums) {
        // padding example
        output_data[i * sample_res_length + offset] = input_data[i];
        lable_data[i * sample_res_length + offset] = 1;
        sampling_depth += 1;
      }
    }  // end all input nce
  }

  std::pair<size_t, size_t> find_leaf_travel(const int64_t &leaf_idx,
                                             const LoD &travel_lod) {
    // check leaf whether in tree
    LoDAndOffset res = framework::GetSubLoDAndAbsoluteOffset(
        travel_lod, leaf_idx, leaf_idx + 1, 0);
    return res.second;
  }
};

}  // namespace operators
}  // namespace paddle
