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

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/inference/capi/c_api.h"
#include "paddle/fluid/inference/capi/c_api_internal.h"

extern "C" {

PD_AnalysisConfig* PD_NewAnalysisConfig() { return new PD_AnalysisConfig; }

void PD_DeleteAnalysisConfig(PD_AnalysisConfig* config) {
  if (config) {
    delete config;
    config = nullptr;
  }
}

void PD_SetModel(PD_AnalysisConfig* config, const char* model_dir,
                 const char* params_path = NULL) {
  PADDLE_ENFORCE(model_dir != nullptr,
                 "Input(model_dir) of PD_SetModel should not be null.");
  if (!params_path) {
    config->config.SetModel(std::string(model_dir));
  } else {
    config->config.SetModel(std::string(model_dir), std::string(params_path));
  }
}

void PD_SetProgFile(PD_AnalysisConfig* config, const char* x) {
  PADDLE_ENFORCE(x != nullptr,
                 "Input(prog_file) of PD_SetProgFile should not be null.");
  config->config.SetProgFile(std::string(x));
}

void PD_SetParamsFile(PD_AnalysisConfig* config, const char* x) {
  PADDLE_ENFORCE(x != nullptr,
                 "Input(params_file) of PD_SetParamsFile should not be null.");
  config->config.SetParamsFile(std::string(x));
}

void PD_SetOptimCacheDir(PD_AnalysisConfig* config, const char* opt_cache_dir) {
  PADDLE_ENFORCE(
      opt_cache_dir != nullptr,
      "Input(opt_cache_dir) of PD_SetOptimCacheDir should not be null.");
  config->config.SetOptimCacheDir(std::string(opt_cache_dir));
}

char* PD_ModelDir(PD_AnalysisConfig* config) {
  return config->config.model_dir().data();
}

char* PD_ProgFile(PD_AnalysisConfig* config) {
  return config->config.prog_file().data();
}

char* PD_ParamsFile(PD_AnalysisConfig* config) {
  return config->config.params_file().data();
}

void PD_EnableUseGpu(PD_AnalysisConfig* config,
                     uint64_t memory_pool_init_size_mb, int device_id = 0) {
  config->config.EnableUseGpu(memory_pool_init_size_mb, device_id);
}

void PD_DisableGpu(PD_AnalysisConfig* config) { config->config.DisableGpu(); }

bool PD_UseGpu(PD_AnalysisConfig* config) { config->config.use_gpu(); }

int PD_GpuDeviceId(PD_AnalysisConfig* config) {
  return config->config.gpu_device_id();
}

int PD_MemoryPoolInitSizeMb(PD_AnalysisConfig* config) {
  return config->config.memory_pool_init_size_mb();
}

float PD_FractionOfGpuMemoryForPool(PD_AnalysisConfig* config) {
  return config->config.fraction_of_gpu_memory_for_pool();
}

void PD_EnableCUDNN(PD_AnalysisConfig* config) { config->config.EnableCUDNN(); }

bool PD_CudnnEnabled(PD_AnalysisConfig* config) {
  return config->config.cudnn_enabled();
}

void PD_SwitchIrOptim(PD_AnalysisConfig* config, int x = true) {
  config->config.SwitchIrOptim(x);
}

bool PD_IrOptim(PD_AnalysisConfig* config) { return config->config.ir_optim(); }

void PD_SwitchUseFeedFetchOps(PD_AnalysisConfig* config, int x = true) {
  config->config.SwitchUseFeedFetchOps(x);
}

bool PD_UseFeedFetchOpsEnabled(PD_AnalysisConfig* config) {
  return config->config.use_feed_fetch_ops_enabled();
}

void PD_SwitchSpecifyInputNames(PD_AnalysisConfig* config, bool x = true) {
  config->config.SwitchSpecifyInputNames(x);
}

bool PD_SpecifyInputName(PD_AnalysisConfig* config) {
  return config->config.specify_input_name();
}

void PD_EnableTensorRtEngine(PD_AnalysisConfig* config,
                             int workspace_size = 1 << 20,
                             int max_batch_size = 1, int min_subgraph_size = 3,
                             Precision precision = Precision::kFloat32,
                             bool use_static = false,
                             bool use_calib_mode = true) {
  config->config.EnableTensorRtEngine(workspace_size, max_batch_size,
                                      min_subgraph_size, precision, use_static,
                                      use_calib_mode);
}

bool PD_TensorrtEngineEnabled(PD_AnalysisConfig* config) {
  return config->config.tensorrt_engine_enabled();
}

void PD_EnableAnakinEngine(PD_AnalysisConfig* config, int max_batch_size = 1,
                           PD_MaxInputShape* max_input_shape = NULL,
                           int max_input_shape_size = 0,
                           int min_subgraph_size = 6,
                           Precision precision = Precision::kFloat32,
                           bool auto_config_layout = false,
                           char** passes_filter = NULL,
                           int passes_filter_size = 0, char** ops_filter = NULL,
                           int ops_filter_size = 0) {
  std::unordered_map<std::string, std::vector<int>> mis;
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
                                    precision, auto_config_layout, pf, of);
}

bool PD_AnakinEngineEnabled(PD_AnalysisConfig* config) {
  return config->config.anakin_engine_enabled();
}

void PD_SwitchIrDebug(PD_AnalysisConfig* config, int x = true) {
  config->config.SwitchIrDebug(x);
}

void PD_EnableNgraph(PD_AnalysisConfig* config) {
  config->config.EnableNgraph();
}

bool PD_NgraphEnabled(PD_AnalysisConfig* config) {
  return config->config.ngraph_enabled();
}

void PD_EnableMKLDNN(PD_AnalysisConfig* config) {
  config->config.EnableMKLDNN();
}

void PD_SetMkldnnCacheCapacity(PD_AnalysisConfig* config, int capacity) {
  config->config.SetMkldnnCacheCapacity(capacity);
}

bool PD_MkldnnEnabled(PD_AnalysisConfig* config) {
  return config->config.mkldnn_enabled();
}

void PD_SetCpuMathLibraryNumThreads(PD_AnalysisConfig* config,
                                    int cpu_math_library_num_threads) {
  config->config.SetCpuMathLibraryNumThreads(cpu_math_library_num_threads);
}

int PD_CpuMathLibraryNumThreads(PD_AnalysisConfig* config) {
  return config->config.cpu_math_library_num_threads();
}

void PD_EnableMkldnnQuantizer(PD_AnalysisConfig* config) {
  config->config.EnableMkldnnQuantizer();
}

bool PD_MkldnnQuantizerEnabled(PD_AnalysisConfig* config) {
  return config->config.mkldnn_quantizer_enabled();
}

void PD_SetModelBuffer(PD_AnalysisConfig* config, const char* prog_buffer,
                       size_t prog_buffer_size, const char* params_buffer,
                       size_t params_buffer_size) {
  config->config.SetModelBuffer(prog_buffer, prog_buffer_size, params_buffer,
                                params_buffer_size);
}

bool PD_ModelFromMemory(PD_AnalysisConfig* config) {
  return config->config.model_from_memory();
}

void PD_EnableMemoryOptim(PD_AnalysisConfig* config, bool static_optim = false,
                          bool force_update_static_cache = false) {
  config->config.enable_memory_optim(static_optim, force_update_static_cache);
}

bool PD_MemoryOptimEnabled(PD_AnalysisConfig* config) {
  return config->config.enable_memory_optim();
}

void PD_EnableProfile(PD_AnalysisConfig* config) {
  config->config.EnableProfile();
}

bool PD_ProfileEnabled(PD_AnalysisConfig* config) {
  return config->config.profile_enabled();
}

void PD_SetInValid(PD_AnalysisConfig* config) { config->config.SetInValid(); }

bool PD_IsValid(PD_AnalysisConfig* config) { return config->config.is_valid(); }
}  // extern "C"
