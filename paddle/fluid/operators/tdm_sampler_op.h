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

#include <gflags/gflags.h>
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
        context.Attr<std::vector<int>>("neg_samples_num_list");
    auto layer_offset_lod = context.Attr<std::vector<int>>("layer_offset_lod");
    auto output_positive_flag = context.Attr<bool>("output_positive");

    // get all tensor
    auto &input_tensor = input_var->Get<framework::LoDTensor>();
    auto &travel_lod_tensor = travel_var->Get<framework::LoDTensor>();
    auto &layer_lod_tensor = layer_var->Get<framework::LoDTensor>();

    // get dimension
    int input_ids_num = input_tensor.numel();
    VLOG(1) << "TDM: input ids nums: " << input_ids_num;
    auto layer_nums = neg_samples_num_vec.size();
    VLOG(1) << "TDM: tree layer nums: " << layer_nums;

    int sample_res_length = 0;
    for (int layer_idx = 0; layer_idx < layer_nums; ++layer_idx) {
      sample_res_length += (neg_samples_num_vec[layer_idx] +
                            static_cast<int>(output_positive_flag));
    }
    VLOG(1) << "TDM: sample res length: " << sample_res_length;

    auto *out_var = context.OutputVar("Out");
    auto *label_var = context.OutputVar("Labels");
    auto *mask_var = context.OutputVar("Mask");

    auto ddim = framework::make_ddim({input_ids_num, sample_res_length});

    auto *out_tensor = out_var->GetMutable<framework::LoDTensor>();
    out_tensor->Resize(ddim);
    auto *label_tensor = label_var->GetMutable<framework::LoDTensor>();
    label_tensor->Resize(ddim);
    auto *mask_tensor = mask_var->GetMutable<framework::LoDTensor>();
    mask_tensor->Resize(ddim);

    // get all data
    auto *input_data = input_tensor.data<int64_t>();
    int *travel_data = const_cast<int *>(travel_lod_tensor.data<int>());
    int *layer_data = const_cast<int *>(layer_lod_tensor.data<int>());

    auto *output_data = out_tensor->mutable_data<int64_t>(context.GetPlace());
    auto *label_data = label_tensor->mutable_data<int64_t>(context.GetPlace());
    auto *mask_data = mask_tensor->mutable_data<int64_t>(context.GetPlace());
    memset(mask_data, 1, sample_res_length * input_ids_num * sizeof(int64_t));

    VLOG(2) << "End get input & output data";
    // generate uniform sampler

    auto seed = context.Attr<int>("seed");
    for (int i = 0; i < input_ids_num; ++i) {
      // find leaf node travel path
      auto input_id = input_data[i];
      VLOG(1) << "TDM: input id: " << input_id;
      auto start_offset = input_id * layer_nums;
      VLOG(1) << "TDM: Start offset(input_id * layer_nums): " << start_offset;
      // nce sample, layer by layer
      int offset = 0;
      for (int layer_idx = 0; layer_idx < layer_nums; ++layer_idx) {
        int sample_num = neg_samples_num_vec[layer_idx];
        VLOG(1) << "TDM: Sample num: " << sample_num;

        int node_nums =
            layer_offset_lod[layer_idx + 1] - layer_offset_lod[layer_idx];
        VLOG(1) << "TDM: layer - " << layer_idx + 1
                << " - has node_nums: " << node_nums;

        if (travel_data[start_offset + layer_idx] == 0) {
          // skip padding
          VLOG(1) << "TDM: Skip padding ";
          for (int i = 0;
               i < sample_num + static_cast<int>(output_positive_flag); i++) {
            output_data[i * sample_res_length + offset] = 0;
            label_data[i * sample_res_length + offset] = 0;
            mask_data[i * sample_res_length + offset] = 0;
            VLOG(1) << "TDM: Res append positive "
                    << output_data[i * sample_res_length + offset];
            VLOG(1) << "TDM: Label append positive "
                    << label_data[i * sample_res_length + offset];
            VLOG(1) << "TDM: Mask append value "
                    << mask_data[i * sample_res_length + offset];
            offset += 1;
          }
          continue;
        }

        Sampler *sampler = new math::UniformSampler(node_nums - 1, seed);
        VLOG(2) << "TDM: get sampler ";

        // If output positive, add itself
        if (output_positive_flag) {
          output_data[i * sample_res_length + offset] =
              travel_data[start_offset + layer_idx];
          label_data[i * sample_res_length + offset] = 1;
          mask_data[i * sample_res_length + offset] = 1;
          VLOG(1) << "TDM: Res append positive "
                  << output_data[i * sample_res_length + offset];
          VLOG(1) << "TDM: Label append positive "
                  << label_data[i * sample_res_length + offset];
          VLOG(1) << "TDM: Mask append value "
                  << mask_data[i * sample_res_length + offset];
          offset += 1;
        }

        // Sampling at layer, until samples enough
        for (int sample_index = 0; sample_index < sample_num; ++sample_index) {
          // Avoid sampling to positive samples
          int64_t sample_res = 0;
          do {
            sample_res = sampler->Sample();
          } while (travel_data[start_offset + layer_idx] ==
                   layer_data[layer_offset_lod[layer_idx] + sample_res]);

          output_data[i * sample_res_length + offset] =
              layer_data[layer_offset_lod[layer_idx] + sample_res];
          label_data[i * sample_res_length + offset] = 0;
          mask_data[i * sample_res_length + offset] = 1;

          VLOG(1) << "TDM: Res append negitive "
                  << output_data[i * sample_res_length + offset];
          VLOG(1) << "TDM: Label append negitive "
                  << label_data[i * sample_res_length + offset];
          VLOG(1) << "TDM: Mask append value "
                  << mask_data[i * sample_res_length + offset];

          offset += 1;
        }  // end layer nce
        delete sampler;
      }  // end one input nce
    }    // end all input nce

    int sample_total_nums = input_ids_num * sample_res_length;
    std::string output_str = "";
    std::string label_str = "";
    std::string mask_str = "";
    for (int i = 0; i < sample_total_nums; i++) {
      output_str += std::to_string(output_data[i]);
      output_str += ",";
      label_str += std::to_string(label_data[i]);
      label_str += ",";
      mask_str += std::to_string(mask_data[i]);
      mask_str += ",";
    }
    VLOG(1) << "TDM: Sample Res " << output_str;
    VLOG(1) << "TDM: Label Res " << label_str;
    VLOG(1) << "TDM: Mask Res " << mask_str;
  }
};

}  // namespace operators
}  // namespace paddle
