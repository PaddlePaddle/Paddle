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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include <paddle_api.h>  // NOLINT
#pragma GCC diagnostic pop

namespace paddle {
namespace inference {
namespace lite {

struct EngineConfig {
  std::string model;
  std::string param;
  std::vector<paddle::lite_api::Place> valid_places;
  std::vector<std::string> neglected_passes;
  lite_api::LiteModelType model_type{lite_api::LiteModelType::kProtobuf};
  bool model_from_memory{true};

  // for xpu
  int xpu_device_id{0};
  size_t xpu_l3_size{0};
  void* xpu_l3_ptr{nullptr};
  size_t xpu_l3_autotune_size{0};
  void* xpu_stream{nullptr};
  int xpu_conv_autotune_level{0};
  std::string xpu_conv_autotune_file;
  bool xpu_conv_autotune_file_writeback{false};
  int xpu_fc_autotune_level{0};
  std::string xpu_fc_autotune_file;
  bool xpu_fc_autotune_file_writeback{false};
  int xpu_gemm_compute_precision{1};
  int xpu_transformer_softmax_optimize_level{0};
  bool xpu_transformer_encoder_adaptive_seqlen{true};
  float xpu_quant_post_static_gelu_out_threshold{10.f};
  int xpu_quant_post_dynamic_activation_method{0};
  bool xpu_enable_multi_stream = false;

  // for x86 or arm
  int cpu_math_library_num_threads{1};

  // for cuda
  bool use_multi_stream{false};

  // for nnadapter or npu.
  std::string nnadapter_model_cache_dir;
  std::vector<std::string> nnadapter_device_names;
  std::string nnadapter_context_properties;
  std::string nnadapter_subgraph_partition_config_buffer;
  std::string nnadapter_subgraph_partition_config_path;
  std::vector<std::string> nnadapter_model_cache_token;
  std::vector<std::vector<char>> nnadapter_model_cache_buffer;

  bool use_opencl{};
  std::string opencl_bin_path = "./";
  std::string opencl_bin_name = "lite_opencl_kernel.bin";
  paddle::lite_api::CLTuneMode opencl_tune_mode{};
  paddle::lite_api::CLPrecisionType opencl_precision_type{};
};

class EngineManager {
 public:
  bool Empty() const;
  bool Has(const std::string& name) const;
  paddle::lite_api::PaddlePredictor* Get(const std::string& name) const;
  paddle::lite_api::PaddlePredictor* Create(const std::string& name,
                                            const EngineConfig& cfg);
  void Set(const std::string& name,
           std::shared_ptr<paddle::lite_api::PaddlePredictor> p);
  void DeleteAll();

 private:
  std::unordered_map<std::string,
                     std::shared_ptr<paddle::lite_api::PaddlePredictor>>
      engines_;
};

}  // namespace lite
}  // namespace inference
}  // namespace paddle
