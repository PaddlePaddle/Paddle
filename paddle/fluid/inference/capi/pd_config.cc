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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/inference/capi/c_api.h"
#include "paddle/fluid/inference/capi/c_api_internal.h"

using paddle::ConvertToPaddleDType;
using paddle::ConvertToPlace;
using paddle::ConvertToPDDataType;
using paddle::ConvertToACPrecision;

extern "C" {

PD_AnalysisConfig* PD_NewAnalysisConfig() { return new PD_AnalysisConfig; }

void PD_DeleteAnalysisConfig(PD_AnalysisConfig* config) {
  if (config) {
    delete config;
    config = nullptr;
  }
}

PD_AnalysisConfig* PD_SetModel(PD_AnalysisConfig* config, const char* model_dir,
                               const char* params_path) {
  // PADDLE_ENFORCE(model_dir != nullptr,
  //                "Input(model_dir) of PD_SetModel should not be null.");
  LOG(INFO) << model_dir;
  LOG(INFO) << std::string(model_dir);
  if (!params_path) {
    config->config.SetModel(std::string(model_dir));
  } else {
    config->config.SetModel(std::string(model_dir), std::string(params_path));
  }
  return config;
}

PD_AnalysisConfig* PD_SetProgFile(PD_AnalysisConfig* config, const char* x) {
  PADDLE_ENFORCE(x != nullptr,
                 "Input(prog_file) of PD_SetProgFile should not be null.");
  config->config.SetProgFile(std::string(x));
  return config;
}

PD_AnalysisConfig* PD_SetParamsFile(PD_AnalysisConfig* config, const char* x) {
  PADDLE_ENFORCE(x != nullptr,
                 "Input(params_file) of PD_SetParamsFile should not be null.");
  config->config.SetParamsFile(std::string(x));
  return config;
}

PD_AnalysisConfig* PD_SetOptimCacheDir(PD_AnalysisConfig* config,
                                       const char* opt_cache_dir) {
  PADDLE_ENFORCE(
      opt_cache_dir != nullptr,
      "Input(opt_cache_dir) of PD_SetOptimCacheDir should not be null.");
  config->config.SetOptimCacheDir(std::string(opt_cache_dir));
  return config;
}

const char* PD_ModelDir(PD_AnalysisConfig* config) {
  return config->config.model_dir().c_str();
}

const char* PD_ProgFile(PD_AnalysisConfig* config) {
  return config->config.prog_file().c_str();
}

const char* PD_ParamsFile(PD_AnalysisConfig* config) {
  return config->config.params_file().c_str();
}

PD_AnalysisConfig* PD_EnableUseGpu(PD_AnalysisConfig* config,
                                   uint64_t memory_pool_init_size_mb,
                                   int device_id) {
  config->config.EnableUseGpu(memory_pool_init_size_mb, device_id);
  return config;
}

PD_AnalysisConfig* PD_DisableGpu(PD_AnalysisConfig* config) {
  config->config.DisableGpu();
  return config;
}

bool PD_UseGpu(PD_AnalysisConfig* config) { return config->config.use_gpu(); }

int PD_GpuDeviceId(PD_AnalysisConfig* config) {
  return config->config.gpu_device_id();
}

int PD_MemoryPoolInitSizeMb(PD_AnalysisConfig* config) {
  return config->config.memory_pool_init_size_mb();
}

float PD_FractionOfGpuMemoryForPool(PD_AnalysisConfig* config) {
  return config->config.fraction_of_gpu_memory_for_pool();
}

PD_AnalysisConfig* PD_EnableCUDNN(PD_AnalysisConfig* config) {
  config->config.EnableCUDNN();
  return config;
}

bool PD_CudnnEnabled(PD_AnalysisConfig* config) {
  return config->config.cudnn_enabled();
}

PD_AnalysisConfig* PD_SwitchIrOptim(PD_AnalysisConfig* config, bool x) {
  config->config.SwitchIrOptim(x);
  return config;
}

bool PD_IrOptim(PD_AnalysisConfig* config) { return config->config.ir_optim(); }

PD_AnalysisConfig* PD_SwitchUseFeedFetchOps(PD_AnalysisConfig* config, bool x) {
  config->config.SwitchUseFeedFetchOps(x);
  return config;
}

bool PD_UseFeedFetchOpsEnabled(PD_AnalysisConfig* config) {
  return config->config.use_feed_fetch_ops_enabled();
}

PD_AnalysisConfig* PD_SwitchSpecifyInputNames(PD_AnalysisConfig* config,
                                              bool x) {
  config->config.SwitchSpecifyInputNames(x);
  return config;
}

bool PD_SpecifyInputName(PD_AnalysisConfig* config) {
  return config->config.specify_input_name();
}

PD_AnalysisConfig* PD_EnableTensorRtEngine(PD_AnalysisConfig* config,
                                           int workspace_size,
                                           int max_batch_size,
                                           int min_subgraph_size,
                                           Precision precision, bool use_static,
                                           bool use_calib_mode) {
  config->config.EnableTensorRtEngine(
      workspace_size, max_batch_size, min_subgraph_size,
      paddle::ConvertToACPrecision(precision), use_static, use_calib_mode);
  return config;
}

bool PD_TensorrtEngineEnabled(PD_AnalysisConfig* config) {
  return config->config.tensorrt_engine_enabled();
}

PD_AnalysisConfig* PD_EnableAnakinEngine(
    PD_AnalysisConfig* config, int max_batch_size,
    PD_MaxInputShape* max_input_shape, int max_input_shape_size,
    int min_subgraph_size, Precision precision, bool auto_config_layout,
    char** passes_filter, int passes_filter_size, char** ops_filter,
    int ops_filter_size) {
  std::map<std::string, std::vector<int>> mis;
  if (max_input_shape) {
    for (int i = 0; i < max_input_shape_size; ++i) {
      std::vector<int> tmp_shape;
      tmp_shape.assign(
          max_input_shape[i].shape,
          max_input_shape[i].shape + max_input_shape[i].shape_size);
      mis[std::string(max_input_shape[i].name)] = std::move(tmp_shape);
    }
  }
  std::vector<std::string> pf;
  std::vector<std::string> of;
  if (passes_filter) {
    pf.assign(passes_filter, passes_filter + passes_filter_size);
  }
  if (ops_filter) {
    of.assign(ops_filter, ops_filter + ops_filter_size);
  }

  config->config.EnableAnakinEngine(max_batch_size, mis, min_subgraph_size,
                                    paddle::ConvertToACPrecision(precision),
                                    auto_config_layout, pf, of);
  return config;
}

bool PD_AnakinEngineEnabled(PD_AnalysisConfig* config) {
  return config->config.anakin_engine_enabled();
}

PD_AnalysisConfig* PD_SwitchIrDebug(PD_AnalysisConfig* config, bool x) {
  config->config.SwitchIrDebug(x);
  return config;
}

PD_AnalysisConfig* PD_EnableNgraph(PD_AnalysisConfig* config) {
  config->config.EnableNgraph();
  return config;
}

bool PD_NgraphEnabled(PD_AnalysisConfig* config) {
  return config->config.ngraph_enabled();
}

PD_AnalysisConfig* PD_EnableMKLDNN(PD_AnalysisConfig* config) {
  config->config.EnableMKLDNN();
  return config;
}

PD_AnalysisConfig* PD_SetMkldnnCacheCapacity(PD_AnalysisConfig* config,
                                             int capacity) {
  config->config.SetMkldnnCacheCapacity(capacity);
  return config;
}

bool PD_MkldnnEnabled(PD_AnalysisConfig* config) {
  return config->config.mkldnn_enabled();
}

PD_AnalysisConfig* PD_SetCpuMathLibraryNumThreads(
    PD_AnalysisConfig* config, int cpu_math_library_num_threads) {
  config->config.SetCpuMathLibraryNumThreads(cpu_math_library_num_threads);
  return config;
}

int PD_CpuMathLibraryNumThreads(PD_AnalysisConfig* config) {
  return config->config.cpu_math_library_num_threads();
}

PD_AnalysisConfig* PD_EnableMkldnnQuantizer(PD_AnalysisConfig* config) {
  config->config.EnableMkldnnQuantizer();
  return config;
}

bool PD_MkldnnQuantizerEnabled(PD_AnalysisConfig* config) {
  return config->config.mkldnn_quantizer_enabled();
}

PD_AnalysisConfig* PD_SetModelBuffer(PD_AnalysisConfig* config,
                                     const char* prog_buffer,
                                     size_t prog_buffer_size,
                                     const char* params_buffer,
                                     size_t params_buffer_size) {
  config->config.SetModelBuffer(prog_buffer, prog_buffer_size, params_buffer,
                                params_buffer_size);
  return config;
}

bool PD_ModelFromMemory(PD_AnalysisConfig* config) {
  return config->config.model_from_memory();
}

PD_AnalysisConfig* PD_EnableMemoryOptim(PD_AnalysisConfig* config,
                                        bool static_optim,
                                        bool force_update_static_cache) {
  config->config.EnableMemoryOptim(static_optim, force_update_static_cache);
  return config;
}

bool PD_MemoryOptimEnabled(PD_AnalysisConfig* config) {
  return config->config.enable_memory_optim();
}

PD_AnalysisConfig* PD_EnableProfile(PD_AnalysisConfig* config) {
  config->config.EnableProfile();
  return config;
}

bool PD_ProfileEnabled(PD_AnalysisConfig* config) {
  return config->config.profile_enabled();
}

PD_AnalysisConfig* PD_SetInValid(PD_AnalysisConfig* config) {
  config->config.SetInValid();
  return config;
}

bool PD_IsValid(PD_AnalysisConfig* config) { return config->config.is_valid(); }
}  // extern "C"
