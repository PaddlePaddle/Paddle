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

#ifdef LITE_SUBGRAPH_WITH_XPU
#define LITE_WITH_XPU 1
#endif

#ifndef PADDLE_WITH_ARM
#define LITE_WITH_X86 1
#endif

#include "paddle/fluid/inference/lite/engine.h"

#include <utility>

namespace paddle {
namespace inference {
namespace lite {

bool EngineManager::Empty() const { return engines_.size() == 0; }

bool EngineManager::Has(const std::string& name) const {
  if (engines_.count(name) == 0) {
    return false;
  }
  return engines_.at(name).get() != nullptr;
}

paddle::lite_api::PaddlePredictor* EngineManager::Get(
    const std::string& name) const {
  return engines_.at(name).get();
}

paddle::lite_api::PaddlePredictor* EngineManager::Create(
    const std::string& name, const EngineConfig& cfg) {
  // config info for predictor.
  paddle::lite_api::CxxConfig lite_cxx_config;
  lite_cxx_config.set_model_buffer(
      cfg.model.c_str(), cfg.model.size(), cfg.param.c_str(), cfg.param.size());
  lite_cxx_config.set_valid_places(cfg.valid_places);
#ifdef PADDLE_WITH_ARM
  lite_cxx_config.set_threads(cfg.cpu_math_library_num_threads);
#else
  lite_cxx_config.set_x86_math_num_threads(cfg.cpu_math_library_num_threads);
#endif

#ifdef LITE_SUBGRAPH_WITH_XPU
  // Deprecated in Paddle-Lite release/v2.8
  lite_cxx_config.set_xpu_workspace_l3_size_per_thread(
      cfg.xpu_l3_workspace_size);
  lite_cxx_config.set_xpu_l3_cache_method(cfg.xpu_l3_workspace_size,
                                          cfg.locked);
  lite_cxx_config.set_xpu_conv_autotune(cfg.autotune, cfg.autotune_file);
  lite_cxx_config.set_xpu_multi_encoder_method(cfg.precision,
                                               cfg.adaptive_seqlen);
  lite_cxx_config.set_xpu_dev_per_thread(cfg.device_id);
  if (cfg.enable_multi_stream) {
    lite_cxx_config.enable_xpu_multi_stream();
  }
#endif

#ifdef LITE_SUBGRAPH_WITH_NPU
  lite_cxx_config.set_nnadapter_device_names(cfg.nnadapter_device_names);
  lite_cxx_config.set_nnadapter_context_properties(
      cfg.nnadapter_context_properties);
  lite_cxx_config.set_nnadapter_model_cache_dir(cfg.nnadapter_model_cache_dir);
  if (!cfg.nnadapter_subgraph_partition_config_path.empty()) {
    lite_cxx_config.set_nnadapter_subgraph_partition_config_path(
        cfg.nnadapter_subgraph_partition_config_path);
  }
  if (!cfg.nnadapter_subgraph_partition_config_buffer.empty()) {
    lite_cxx_config.set_nnadapter_subgraph_partition_config_buffer(
        cfg.nnadapter_subgraph_partition_config_buffer);
  }
  for (size_t i = 0; i < cfg.nnadapter_model_cache_token.size(); ++i) {
    lite_cxx_config.set_nnadapter_model_cache_buffers(
        cfg.nnadapter_model_cache_token[i],
        cfg.nnadapter_model_cache_buffer[i]);
  }
#endif

  if (cfg.use_opencl) {
    lite_cxx_config.set_opencl_binary_path_name(cfg.opencl_bin_path,
                                                cfg.opencl_bin_name);
    lite_cxx_config.set_opencl_tune(cfg.opencl_tune_mode);
    lite_cxx_config.set_opencl_precision(cfg.opencl_precision_type);
  }

  // create predictor
  std::shared_ptr<paddle::lite_api::PaddlePredictor> p =
      paddle::lite_api::CreatePaddlePredictor(lite_cxx_config);
  engines_[name] = std::move(p);
  return engines_[name].get();
}

void EngineManager::DeleteAll() {
  for (auto& item : engines_) {
    item.second.reset();
  }
}

}  // namespace lite
}  // namespace inference
}  // namespace paddle
