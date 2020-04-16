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
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/fleet/kv_maps.h"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/sampler.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using Sampler = math::Sampler;
using DDim = framework::DDim;
using LoD = framework::LoD;
using LoDTensor = framework::LoDTensor;
using LoDAndOffset = std::pair<LoD, std::pair<size_t, size_t>>;

template <typename T, typename TreeT = int, typename OutT = int>
void TDMSamplerInner(const framework::ExecutionContext &context,
                     const LoDTensor &input_tensor,
                     std::shared_ptr<framework::UUMAP> travel_info,
                     const LoDTensor &layer_lod_tensor, LoDTensor *out_tensor,
                     LoDTensor *label_tensor, LoDTensor *mask_tensor) {
  auto neg_samples_num_vec =
      context.Attr<std::vector<int>>("neg_samples_num_list");
  auto layer_offset_lod = context.Attr<std::vector<int>>("layer_offset_lod");
  auto output_positive_flag = context.Attr<bool>("output_positive");

  // get dimension
  int input_ids_num = input_tensor.numel();
  VLOG(1) << "TDM: input ids nums: " << input_ids_num;
  auto layer_nums = neg_samples_num_vec.size();
  VLOG(1) << "TDM: tree layer nums: " << layer_nums;

  int sample_res_length = 0;
  for (size_t layer_idx = 0; layer_idx < layer_nums; ++layer_idx) {
    sample_res_length += (neg_samples_num_vec[layer_idx] +
                          static_cast<int>(output_positive_flag));
  }
  VLOG(1) << "TDM: sample res length: " << sample_res_length;

  auto total_sample_nums = input_ids_num * sample_res_length;

  // get all data
  auto *input_data = input_tensor.data<T>();
  auto *layer_data = layer_lod_tensor.data<TreeT>();

  OutT zero = 0;
  OutT one = 1;
  std::vector<OutT> output_vec(total_sample_nums, zero);
  std::vector<OutT> label_vec(total_sample_nums, zero);
  std::vector<OutT> mask_vec(total_sample_nums, one);

  VLOG(1) << "End get input & output data";
  // generate uniform sampler

  auto seed = context.Attr<int>("seed");
  std::vector<Sampler *> sampler_vec{};
  for (size_t layer_index = 0; layer_index < layer_nums; layer_index++) {
    int layer_node_nums =
        layer_offset_lod[layer_index + 1] - layer_offset_lod[layer_index];
    Sampler *sampler = new math::UniformSampler(layer_node_nums - 1, seed);
    sampler_vec.push_back(sampler);
  }
  VLOG(1) << "TDM: get sampler ";

  for (int i = 0; i < input_ids_num; ++i) {
    // find leaf node travel path
    T input_id = input_data[i];

    VLOG(1) << "TDM: input id: " << input_id;
    std::vector<int64_t> travel_data(layer_nums, 0);
    if (travel_info->find(input_id) != travel_info->end())
      travel_data = travel_info->at(input_id);
    
    // nce sample, layer by layer
    int offset = 0;
    for (size_t layer_idx = 0; layer_idx < layer_nums; ++layer_idx) {
      int sample_num = neg_samples_num_vec[layer_idx];
      VLOG(1) << "TDM: Sample num: " << sample_num;

      int node_nums =
          layer_offset_lod[layer_idx + 1] - layer_offset_lod[layer_idx];
      VLOG(1) << "TDM: layer - " << layer_idx + 1
              << " - has node_nums: " << node_nums;

      PADDLE_ENFORCE_LE(
          sample_num, node_nums - 1,
          "Neg sample nums id of OP(fluid.layers.tdm_sampler) at layer %ld "
          "expected <= %ld - 1 (positive included), but got %ld. Please "
          "check neg_samples_num_list.",
          layer_idx, node_nums, sample_num);

      int64_t positive_node_id = travel_data[layer_idx];

      if (positive_node_id == 0) {
        // skip padding
        VLOG(1) << "TDM: Skip padding ";
        for (int sample_index = 0;
             sample_index < sample_num + static_cast<int>(output_positive_flag);
             sample_index++) {
          output_vec[i * sample_res_length + offset] = 0;
          label_vec[i * sample_res_length + offset] = 0;
          mask_vec[i * sample_res_length + offset] = 0;
          VLOG(1) << "TDM: Res append positive "
                  << output_vec[i * sample_res_length + offset]
                  << " Label append positive "
                  << label_vec[i * sample_res_length + offset]
                  << " Mask append value "
                  << mask_vec[i * sample_res_length + offset];
          offset += 1;
        }
        continue;
      }


      // If output positive, add itself
      if (output_positive_flag) {
        output_vec[i * sample_res_length + offset] = static_cast<OutT>(positive_node_id);
        label_vec[i * sample_res_length + offset] = 1;
        mask_vec[i * sample_res_length + offset] = 1;
        VLOG(1) << "TDM: node id: " << positive_node_id << " Res append  "
                << output_vec[i * sample_res_length + offset]
                << " Label append  "
                << label_vec[i * sample_res_length + offset] << " Mask append  "
                << mask_vec[i * sample_res_length + offset];
        offset += 1;
      }
      std::vector<int> sample_res_vec{};
      // Sampling at layer, until samples enough
      for (int sample_index = 0; sample_index < sample_num; ++sample_index) {
        // Avoid sampling to positive samples
        int sample_res = 0;
        do {
          sample_res = sampler_vec[layer_idx]->Sample();
        } while (positive_node_id ==
                     layer_data[layer_offset_lod[layer_idx] + sample_res] ||
                 find(sample_res_vec.begin(), sample_res_vec.end(),
                      sample_res) != sample_res_vec.end());
        sample_res_vec.push_back(sample_res);

        output_vec[i * sample_res_length + offset] = static_cast<OutT>(
            layer_data[layer_offset_lod[layer_idx] + sample_res]);
        label_vec[i * sample_res_length + offset] = 0;
        mask_vec[i * sample_res_length + offset] = 1;
        VLOG(1) << "TDM: node id: " << positive_node_id
                << " Res append negitive "
                << output_vec[i * sample_res_length + offset]
                << " Label append negitive "
                << label_vec[i * sample_res_length + offset]
                << " Mask append value "
                << mask_vec[i * sample_res_length + offset];

        offset += 1;
      }  // end layer nce
    }    // end one input nce
  }      // end all input nce

  auto *output_data = out_tensor->mutable_data<OutT>(context.GetPlace());
  auto *label_data = label_tensor->mutable_data<OutT>(context.GetPlace());
  auto *mask_data = mask_tensor->mutable_data<OutT>(context.GetPlace());

  memcpy(output_data, &output_vec[0], sizeof(OutT) * total_sample_nums);
  memcpy(label_data, &label_vec[0], sizeof(OutT) * total_sample_nums);
  memcpy(mask_data, &mask_vec[0], sizeof(OutT) * total_sample_nums);

  for (size_t layer_index = 0; layer_index < layer_nums; layer_index++) {
    delete sampler_vec[layer_index];
  }
}

template <typename DeviceContext, typename T>
class TDMSamplerKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *input_var = context.InputVar("X");
    auto *layer_var = context.InputVar("Layer");

    // get all tensor
    auto &input_tensor = input_var->Get<framework::LoDTensor>();
   // auto &travel_lod_tensor = travel_var->Get<framework::LoDTensor>();
    std::shared_ptr<framework::UUMAP> travel_info =
        framework::KV_MAPS::GetInstance()->get_data("travel_info");
    for (auto ite = travel_info->begin(); ite != travel_info->end(); ite++) {
        VLOG(1) << ite->first << " " << ite->second[0];
    }
    auto &layer_lod_tensor = layer_var->Get<framework::LoDTensor>();

    const auto &input_type = input_tensor.type();
    bool input_type_match = input_type == framework::proto::VarType::INT32 ||
                            input_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(input_type_match, true,
                      platform::errors::InvalidArgument(
                          "Input(X) holds the wrong type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(input_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    const auto &layer_type = layer_lod_tensor.type();
    bool layer_type_match = layer_type == framework::proto::VarType::INT32 ||
                            layer_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(layer_type_match, true,
                      platform::errors::InvalidArgument(
                          "Input(Layer) holds the wrong type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(layer_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    auto *out_var = context.OutputVar("Out");
    auto *label_var = context.OutputVar("Labels");
    auto *mask_var = context.OutputVar("Mask");
    auto *out_tensor = out_var->GetMutable<framework::LoDTensor>();
    auto *label_tensor = label_var->GetMutable<framework::LoDTensor>();
    auto *mask_tensor = mask_var->GetMutable<framework::LoDTensor>();

    auto output_type = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("dtype"));

    if (output_type == framework::proto::VarType::INT32) {
      TDMSamplerInner<T, int64_t, int>(context, input_tensor, travel_info,
                                   layer_lod_tensor, out_tensor, label_tensor,
                                   mask_tensor);
    } else if (output_type == framework::proto::VarType::INT64) {
      TDMSamplerInner<T, int64_t, int64_t>(
          context, input_tensor, travel_info, layer_lod_tensor,
          out_tensor, label_tensor, mask_tensor);
    }
  }
};

}  // namespace operators
}  // namespace paddle
