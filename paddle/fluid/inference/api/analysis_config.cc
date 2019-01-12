// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <boost/variant.hpp>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {

enum class ConfigDType {
  kInt = 0,
  kBool,
  kString,
  kFloat,
};
using any_value_t =
    std::pair<boost::variant<bool, int, std::string, float>, ConfigDType>;
using extra_config_t = std::unordered_map<std::string, any_value_t>;
template <typename T>
ConfigDType GetConfigDType();
template <ConfigDType dtype>
void SetExtraConfigValue(const any_value_t &x, const std::string &key,
                         extra_config_t *config);

template <>
ConfigDType GetConfigDType<int>() {
  return ConfigDType::kInt;
}
template <>
ConfigDType GetConfigDType<bool>() {
  return ConfigDType::kBool;
}

template <>
ConfigDType GetConfigDType<std::string>() {
  return ConfigDType::kString;
}
template <>
ConfigDType GetConfigDType<float>() {
  return ConfigDType::kFloat;
}

#define IMPL_SET_EXTRA_CONFIG_VALUE(D, T)                                     \
  template <>                                                                 \
  void SetExtraConfigValue<ConfigDType::D>(                                   \
      const any_value_t &x, const std::string &key, extra_config_t *config) { \
    config->emplace(key, std::make_pair(boost::get<T>(x.first), x.second));   \
  }
IMPL_SET_EXTRA_CONFIG_VALUE(kString, std::string);
IMPL_SET_EXTRA_CONFIG_VALUE(kInt, int);
IMPL_SET_EXTRA_CONFIG_VALUE(kFloat, float);
IMPL_SET_EXTRA_CONFIG_VALUE(kBool, bool);

PassStrategy *contrib::AnalysisConfig::pass_builder() const {
  if (!pass_builder_.get()) {
    if (use_gpu_) {
      LOG(INFO) << "Create GPU IR passes";
      pass_builder_.reset(new GpuPassStrategy);
    } else {
      LOG(INFO) << "Create CPU IR passes";
      pass_builder_.reset(new CpuPassStrategy);
    }
  } else if (pass_builder_->use_gpu() ^ use_gpu()) {
    LOG(WARNING) << "The use_gpu flag is not compatible between Config and "
                    "PassBuilder, the flags are "
                 << use_gpu() << " " << pass_builder_->use_gpu();
    LOG(WARNING) << "Please make them compatible, still use the existing "
                    "PassBuilder.";
  }

  return pass_builder_.get();
}

contrib::AnalysisConfig::AnalysisConfig(const std::string &model_dir) {
  model_dir_ = model_dir;

  Update();
}
contrib::AnalysisConfig::AnalysisConfig(const std::string &prog_file,
                                        const std::string &params_file) {
  prog_file_ = prog_file;
  params_file_ = params_file;

  Update();
}
void contrib::AnalysisConfig::SetModel(const std::string &prog_file_path,
                                       const std::string &params_file_path) {
  prog_file_ = prog_file_path;
  params_file_ = params_file_path;

  Update();
}
void contrib::AnalysisConfig::EnableUseGpu(uint64_t memory_pool_init_size_mb,
                                           int device_id) {
#ifdef PADDLE_WITH_CUDA
  use_gpu_ = true;
  memory_pool_init_size_mb_ = memory_pool_init_size_mb;
  device_id_ = device_id;
#else
  LOG(ERROR) << "Please compile with gpu to EnableGpu";
  use_gpu_ = false;
#endif

  Update();
}
void contrib::AnalysisConfig::DisableGpu() {
  use_gpu_ = false;

  Update();
}

void contrib::AnalysisConfig::EnableMKLDNN() {
#ifdef PADDLE_WITH_MKLDNN
  pass_builder()->EnableMKLDNN();
  use_mkldnn_ = true;
#else
  LOG(ERROR) << "Please compile with MKLDNN first to use MKLDNN";
  use_mkldnn_ = false;
#endif

  Update();
}

void contrib::AnalysisConfig::EnableTensorRtEngine(int workspace_size,
                                                   int max_batch_size,
                                                   int min_subgraph_size) {
  use_tensorrt_ = true;
  tensorrt_workspace_size_ = workspace_size;
  tensorrt_max_batchsize_ = max_batch_size;
  tensorrt_min_subgraph_size_ = min_subgraph_size;

  Update();
}

void contrib::AnalysisConfig::Update() {
  auto info = SerializeInfoCache();
  if (info == serialized_info_cache_) return;

  if (use_gpu_) {
    pass_builder_.reset(new GpuPassStrategy);
  } else {
    pass_builder_.reset(new CpuPassStrategy);
  }

  if (use_tensorrt_) {
    if (!use_gpu_) {
      LOG(ERROR)
          << "TensorRT engine is not available when EnableGpu() not actived.";
    } else {
      // Append after the infer_clean pass.
      pass_builder()->InsertPass(1, "tensorrt_subgraph_pass");
    }
  }

  if (use_mkldnn_) {
    if (!enable_ir_optim_) {
      LOG(ERROR)
          << "EnableMKLDNN() only works when IR optimization is enabled.";
    }
#ifdef PADDLE_WITH_MKLDNN
    pass_builder()->EnableMKLDNN();
    use_mkldnn_ = true;
#else
    LOG(ERROR) << "Please compile with MKLDNN first to use MKLDNN";
    use_mkldnn_ = false;
#endif
  }

  if (ir_debug_) {
    pass_builder()->TurnOnDebug();
  }
}

std::string contrib::AnalysisConfig::SerializeInfoCache() {
  std::stringstream ss;
  ss << model_dir_;
  ss << prog_file_;
  ss << params_file_;

  ss << use_gpu_;
  ss << device_id_;
  ss << memory_pool_init_size_mb_;

  ss << use_tensorrt_;
  ss << tensorrt_workspace_size_;
  ss << tensorrt_max_batchsize_;
  ss << tensorrt_min_subgraph_size_;

  ss << use_mkldnn_;
  for (auto &item : mkldnn_enabled_op_types_) ss << item;

  ss << model_from_memory_;
  ss << enable_ir_optim_;
  ss << use_feed_fetch_ops_;
  ss << ir_debug_;

  ss << specify_input_name_;
  ss << cpu_math_library_num_threads_;

  return ss.str();
}

void contrib::AnalysisConfig::SetCpuMathLibraryNumThreads(
    int cpu_math_library_num_threads) {
  cpu_math_library_num_threads_ = cpu_math_library_num_threads;

  Update();
}

float contrib::AnalysisConfig::fraction_of_gpu_memory_for_pool() const {
#ifdef PADDLE_WITH_CUDA
  // Get the GPU memory details and calculate the fraction of memory for the
  // GPU memory pool.
  size_t gpu_used, gpu_available;
  platform::GpuMemoryUsage(&gpu_used, &gpu_available);
  double total_gpu_memory = (gpu_used + gpu_available) / 1024. / 1024.;
  float fraction_of_gpu_memory =
      static_cast<double>(memory_pool_init_size_mb()) / total_gpu_memory;
  return fraction_of_gpu_memory;
#else
  return 0.;
#endif
}

void contrib::AnalysisConfig::SetModelBuffer(const char *prog_buffer,
                                             size_t prog_buffer_size,
                                             const char *param_buffer,
                                             size_t param_buffer_size) {
  prog_file_ = std::string(prog_buffer, prog_buffer + prog_buffer_size);
  params_file_ = std::string(param_buffer, param_buffer + param_buffer_size);
  model_from_memory_ = true;

  Update();
}

contrib::AnalysisConfig::~AnalysisConfig() {
  if (extra_configs_) {
    delete static_cast<std::unordered_map<std::string, any_value_t> *>(
        extra_configs_);
  }
}

namespace contrib {

#define GET_EXTRA_CONFIG                                               \
  if (!extra_configs_)                                                 \
    extra_configs_ = new std::unordered_map<std::string, any_value_t>; \
  PADDLE_ENFORCE_NOT_NULL(extra_configs_);                             \
  auto *extra_config = static_cast<extra_config_t *>(extra_configs_);

template <typename T>
void contrib::AnalysisConfig::SetExtraConfig(const std::string &key,
                                             const T &value) {
  GET_EXTRA_CONFIG;
  auto it = extra_config->find(key);
  if (it == extra_config->end()) {
    extra_config->emplace(key, std::make_pair(value, GetConfigDType<T>()));
  } else {
    it->second = std::make_pair(value, GetConfigDType<T>());
  }
}

template <typename T>
T contrib::AnalysisConfig::extra_config(const std::string &key) {
  GET_EXTRA_CONFIG;
  auto it = extra_config->find(key);
  PADDLE_ENFORCE(it != extra_config->end(), "no config [%s}", key);
  return boost::get<T>(it->second.first);
}

bool contrib::AnalysisConfig::HasExtraConfig(const std::string &key) {
  GET_EXTRA_CONFIG;
  return extra_config->find(key) != extra_config->end();
}

#define IMPL_EXTRA_CONFIG(T)                             \
  template void contrib::AnalysisConfig::SetExtraConfig( \
      const std::string &key, const T &value);           \
  template T contrib::AnalysisConfig::extra_config(const std::string &key);

IMPL_EXTRA_CONFIG(int);
IMPL_EXTRA_CONFIG(float);
IMPL_EXTRA_CONFIG(bool);
IMPL_EXTRA_CONFIG(std::string);

contrib::AnalysisConfig::AnalysisConfig(const contrib::AnalysisConfig &other) {
#define CP_MEMBER(member__) member__ = other.member__;

  // Model related.
  CP_MEMBER(model_dir_);
  CP_MEMBER(prog_file_);
  CP_MEMBER(params_file_);
  CP_MEMBER(model_from_memory_);  // the memory model reuses prog_file_ and
  // params_file_ fields.
  // Gpu related.
  CP_MEMBER(use_gpu_);
  CP_MEMBER(device_id_);
  CP_MEMBER(memory_pool_init_size_mb_);
  // TensorRT related.
  CP_MEMBER(use_tensorrt_);
  CP_MEMBER(tensorrt_workspace_size_);
  CP_MEMBER(tensorrt_max_batchsize_);
  CP_MEMBER(tensorrt_min_subgraph_size_);
  // MKLDNN related.
  CP_MEMBER(use_mkldnn_);
  CP_MEMBER(mkldnn_enabled_op_types_);

  // Ir related.
  CP_MEMBER(enable_ir_optim_);
  CP_MEMBER(use_feed_fetch_ops_);
  CP_MEMBER(ir_debug_);
  CP_MEMBER(specify_input_name_);

  CP_MEMBER(cpu_math_library_num_threads_);

  CP_MEMBER(serialized_info_cache_);

  if (use_gpu_) {
    pass_builder_.reset(new GpuPassStrategy(
        *static_cast<GpuPassStrategy *>(other.pass_builder())));
  } else {
    pass_builder_.reset(new CpuPassStrategy(
        *static_cast<CpuPassStrategy *>(other.pass_builder())));
  }

  if (other.extra_configs_) {
    GET_EXTRA_CONFIG;
    auto *other_extra_config =
        static_cast<extra_config_t *>(other.extra_configs_);
    for (auto &item : *other_extra_config) {
      switch (item.second.second) {
        case ConfigDType::kBool:
          SetExtraConfigValue<ConfigDType::kBool>(item.second, item.first,
                                                  extra_config);
          break;
        case ConfigDType::kFloat:
          SetExtraConfigValue<ConfigDType::kFloat>(item.second, item.first,
                                                   extra_config);
          break;
        case ConfigDType::kInt:
          SetExtraConfigValue<ConfigDType::kInt>(item.second, item.first,
                                                 extra_config);
          break;
        case ConfigDType::kString:
          SetExtraConfigValue<ConfigDType::kString>(item.second, item.first,
                                                    extra_config);
          break;
      }
    }
  }

#undef CP_MEMBER

  Update();
}

}  // namespace contrib

}  // namespace paddle
