// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <cmath>
#include <vector>
#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/math/sampler.h"

namespace phi {

using Sampler = math::Sampler;

template <typename T,
          typename Context,
          typename TreeT = int,
          typename OutT = int>
void TDMSamplerInner(const Context &dev_ctx,
                     const phi::DenseTensor &input_tensor,
                     const phi::DenseTensor &travel_dense_tensor,
                     const phi::DenseTensor &layer_dense_tensor,
                     bool output_positive,
                     std::vector<int> neg_samples_num_list,
                     std::vector<int> layer_offset,
                     int seed,
                     phi::DenseTensor *out,
                     phi::DenseTensor *label,
                     phi::DenseTensor *mask) {
  // get dimension
  int input_ids_num = input_tensor.numel();
  VLOG(3) << "TDM: input ids nums: " << input_ids_num;
  auto layer_nums = neg_samples_num_list.size();
  VLOG(3) << "TDM: tree layer nums: " << layer_nums;

  int sample_res_length = 0;
  for (size_t layer_idx = 0; layer_idx < layer_nums; ++layer_idx) {
    sample_res_length +=
        (neg_samples_num_list[layer_idx] + static_cast<int>(output_positive));
  }
  VLOG(3) << "TDM: sample res length: " << sample_res_length;

  auto travel_dim = common::vectorize<int>(travel_dense_tensor.dims());
  auto total_sample_nums = input_ids_num * sample_res_length;

  // get all data
  auto *input_data = input_tensor.data<T>();
  auto *travel_data = travel_dense_tensor.data<TreeT>();
  auto *layer_data = layer_dense_tensor.data<TreeT>();

  OutT zero = 0;
  OutT one = 1;
  std::vector<OutT> output_vec(total_sample_nums, zero);
  std::vector<OutT> label_vec(total_sample_nums, zero);
  std::vector<OutT> mask_vec(total_sample_nums, one);

  VLOG(3) << "End get input & output data";
  // generate uniform sampler

  std::vector<Sampler *> sampler_vec{};
  for (size_t layer_index = 0; layer_index < layer_nums; layer_index++) {
    int layer_node_nums =
        layer_offset[layer_index + 1] - layer_offset[layer_index];
    Sampler *sampler = new math::UniformSampler(layer_node_nums - 1, seed);
    sampler_vec.push_back(sampler);
  }
  VLOG(3) << "TDM: get sampler ";

  for (int i = 0; i < input_ids_num; ++i) {
    // find leaf node travel path
    T input_id = input_data[i];
    PADDLE_ENFORCE_LT(
        -1,
        input_id,
        common::errors::InvalidArgument(
            "Variable value (input) of OP(fluid.layers.tdm_sampler) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            travel_dim[0],
            input_id));
    PADDLE_ENFORCE_LT(
        input_id,
        travel_dim[0],
        common::errors::InvalidArgument(
            "Variable value (input) of OP(fluid.layers.tdm_sampler) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            travel_dim[0],
            input_id));

    VLOG(3) << "TDM: input id: " << input_id;
    int start_offset = static_cast<int>(input_id * layer_nums);
    VLOG(3) << "TDM: Start offset(input_id * layer_nums): " << start_offset;
    // nce sample, layer by layer
    int offset = 0;
    for (size_t layer_idx = 0; layer_idx < layer_nums; ++layer_idx) {
      int sample_num = neg_samples_num_list[layer_idx];
      VLOG(3) << "TDM: Sample num: " << sample_num;

      int node_nums = layer_offset[layer_idx + 1] - layer_offset[layer_idx];
      VLOG(3) << "TDM: layer - " << layer_idx + 1
              << " - has node_nums: " << node_nums;

      PADDLE_ENFORCE_LE(
          sample_num,
          node_nums - 1,
          common::errors::InvalidArgument(
              "Neg sample nums id of OP(fluid.layers.tdm_sampler) at layer %ld "
              "expected <= %ld - 1 (positive included), but got %ld. Please "
              "check neg_samples_num_list.",
              layer_idx,
              node_nums,
              sample_num));

      int node_id_min = layer_offset[layer_idx];
      int node_id_max = layer_offset[layer_idx + 1];

      OutT positive_node_id =
          static_cast<OutT>(travel_data[start_offset + layer_idx]);

      if (positive_node_id == 0) {
        // skip padding
        VLOG(3) << "TDM: Skip padding ";
        for (int sample_index = 0;
             sample_index < sample_num + static_cast<int>(output_positive);
             sample_index++) {
          output_vec[i * sample_res_length + offset] = 0;
          label_vec[i * sample_res_length + offset] = 0;
          mask_vec[i * sample_res_length + offset] = 0;
          VLOG(3) << "TDM: Res append positive "
                  << output_vec[i * sample_res_length + offset]
                  << " Label append positive "
                  << label_vec[i * sample_res_length + offset]
                  << " Mask append value "
                  << mask_vec[i * sample_res_length + offset];
          offset += 1;
        }
        continue;
      }

      PADDLE_ENFORCE_LE(
          positive_node_id,
          node_id_max,
          common::errors::InvalidArgument(
              "Positive node id of OP(fluid.layers.tdm_sampler) at layer %ld "
              "expected >= %ld and <= %ld, but got %ld. Please check input "
              "value.",
              layer_idx,
              node_id_min,
              node_id_max,
              positive_node_id));
      PADDLE_ENFORCE_LE(
          node_id_min,
          positive_node_id,
          common::errors::InvalidArgument(
              "Positive node id of OP(fluid.layers.tdm_sampler) at layer %ld "
              "expected >= %ld and <= %ld, but got %ld. Please check input "
              "value.",
              layer_idx,
              node_id_min,
              node_id_max,
              positive_node_id));

      // If output positive, add itself
      if (output_positive) {
        output_vec[i * sample_res_length + offset] = positive_node_id;
        label_vec[i * sample_res_length + offset] = 1;
        mask_vec[i * sample_res_length + offset] = 1;
        VLOG(3) << "TDM: node id: " << positive_node_id << " Res append  "
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
                     layer_data[layer_offset[layer_idx] + sample_res] ||
                 find(sample_res_vec.begin(),
                      sample_res_vec.end(),
                      sample_res) != sample_res_vec.end());
        sample_res_vec.push_back(sample_res);

        output_vec[i * sample_res_length + offset] =
            static_cast<OutT>(layer_data[layer_offset[layer_idx] + sample_res]);
        label_vec[i * sample_res_length + offset] = 0;
        mask_vec[i * sample_res_length + offset] = 1;
        VLOG(3) << "TDM: node id: " << travel_data[start_offset + layer_idx]
                << " Res append negative "
                << output_vec[i * sample_res_length + offset]
                << " Label append negative "
                << label_vec[i * sample_res_length + offset]
                << " Mask append value "
                << mask_vec[i * sample_res_length + offset];

        PADDLE_ENFORCE_LE(
            layer_data[layer_offset[layer_idx] + sample_res],
            node_id_max,
            common::errors::InvalidArgument(
                "Negative node id of OP(fluid.layers.tdm_sampler) at layer %ld"
                "expected >= %ld and <= %ld, but got %ld. Please check input "
                "tdm tree structure and tdm travel info.",
                layer_idx,
                node_id_min,
                node_id_max,
                layer_data[layer_offset[layer_idx] + sample_res]));

        offset += 1;
      }  // end layer nce
    }    // end one input nce
  }      // end all input nce

  auto *output_data = dev_ctx.template Alloc<OutT>(out);
  auto *label_data = dev_ctx.template Alloc<OutT>(label);
  auto *mask_data = dev_ctx.template Alloc<OutT>(mask);

  memcpy(output_data, &output_vec[0], sizeof(OutT) * total_sample_nums);
  memcpy(label_data, &label_vec[0], sizeof(OutT) * total_sample_nums);
  memcpy(mask_data, &mask_vec[0], sizeof(OutT) * total_sample_nums);

  for (size_t layer_index = 0; layer_index < layer_nums; layer_index++) {
    delete sampler_vec[layer_index];
  }
}

template <typename T, typename Context>
void TDMSamplerKernel(const Context &dev_ctx,
                      const DenseTensor &x,
                      const DenseTensor &travel,
                      const DenseTensor &layer,
                      bool output_positive,
                      const std::vector<int> &neg_samples_num_list,
                      const std::vector<int> &layer_offset,
                      int seed,
                      int dtype,
                      DenseTensor *out,
                      DenseTensor *labels,
                      DenseTensor *mask) {
  const auto &input_type = phi::TransToProtoVarType(x.dtype());
  bool input_type_match =
      input_type == ProtoDataType::INT32 || input_type == ProtoDataType::INT64;
  PADDLE_ENFORCE_EQ(input_type_match,
                    true,
                    common::errors::InvalidArgument(
                        "Input(X) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        phi::DataTypeToString(x.dtype()),
                        phi::DataTypeToString(DataType::INT32),
                        phi::DataTypeToString(DataType::INT64)));

  const auto &travel_type = phi::TransToProtoVarType(travel.dtype());
  bool travel_type_match = travel_type == ProtoDataType::INT32 ||
                           travel_type == ProtoDataType::INT64;
  PADDLE_ENFORCE_EQ(travel_type_match,
                    true,
                    common::errors::InvalidArgument(
                        "Input(Travel) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        phi::DataTypeToString(travel.dtype()),
                        phi::DataTypeToString(DataType::INT32),
                        phi::DataTypeToString(DataType::INT64)));

  const auto &layer_type = phi::TransToProtoVarType(layer.dtype());
  bool layer_type_match =
      layer_type == ProtoDataType::INT32 || layer_type == ProtoDataType::INT64;
  PADDLE_ENFORCE_EQ(layer_type_match,
                    true,
                    common::errors::InvalidArgument(
                        "Input(Layer) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        phi::DataTypeToString(layer.dtype()),
                        phi::DataTypeToString(DataType::INT32),
                        phi::DataTypeToString(DataType::INT64)));
  PADDLE_ENFORCE_EQ(travel_type,
                    layer_type,
                    common::errors::InvalidArgument(
                        "Input(Travel) must holds the same type with "
                        "Input(Layer), but Travel holds %s, and Layer holds %s",
                        phi::DataTypeToString(travel.dtype()),
                        phi::DataTypeToString(layer.dtype())));

  auto output_type = static_cast<ProtoDataType>(dtype);

  if (travel_type == ProtoDataType::INT32 &&
      output_type == ProtoDataType::INT32) {
    TDMSamplerInner<T, Context, int, int>(dev_ctx,
                                          x,
                                          travel,
                                          layer,
                                          output_positive,
                                          neg_samples_num_list,
                                          layer_offset,
                                          seed,
                                          out,
                                          labels,
                                          mask);
  } else if (travel_type == ProtoDataType::INT64 &&
             output_type == ProtoDataType::INT32) {
    TDMSamplerInner<T, Context, int64_t, int>(dev_ctx,
                                              x,
                                              travel,
                                              layer,
                                              output_positive,
                                              neg_samples_num_list,
                                              layer_offset,
                                              seed,
                                              out,
                                              labels,
                                              mask);
  } else if (travel_type == ProtoDataType::INT32 &&
             output_type == ProtoDataType::INT64) {
    TDMSamplerInner<T, Context, int, int64_t>(dev_ctx,
                                              x,
                                              travel,
                                              layer,
                                              output_positive,
                                              neg_samples_num_list,
                                              layer_offset,
                                              seed,
                                              out,
                                              labels,
                                              mask);
  } else if (travel_type == ProtoDataType::INT64 &&
             output_type == ProtoDataType::INT64) {
    TDMSamplerInner<T, Context, int64_t, int64_t>(dev_ctx,
                                                  x,
                                                  travel,
                                                  layer,
                                                  output_positive,
                                                  neg_samples_num_list,
                                                  layer_offset,
                                                  seed,
                                                  out,
                                                  labels,
                                                  mask);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(tdm_sampler,
                   CPU,
                   ALL_LAYOUT,
                   phi::TDMSamplerKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
