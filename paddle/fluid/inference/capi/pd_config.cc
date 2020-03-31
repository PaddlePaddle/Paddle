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
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/inference/capi/c_api_internal.h"
#include "paddle/fluid/inference/capi/paddle_c_api.h"

using paddle::ConvertToACPrecision;
using paddle::ConvertToPaddleDType;
using paddle::ConvertToPDDataType;

extern "C" {

PD_AnalysisConfig* PD_NewAnalysisConfig() { return new PD_AnalysisConfig; }  //

void PD_DeleteAnalysisConfig(PD_AnalysisConfig* config) {
  if (config) {
    delete config;
    config = nullptr;
    VLOG(3) << "PD_AnalysisConfig delete successfully. ";
  }
}

void PD_SetModel(PD_AnalysisConfig* config, const char* model_dir,
                 const char* params_path) {
  LOG(INFO) << model_dir;
  PADDLE_ENFORCE_NOT_NULL(config);
  LOG(INFO) << std::string(model_dir);
  if (!params_path) {
    config->config.SetModel(std::string(model_dir));
  } else {
    config->config.SetModel(std::string(model_dir), std::string(params_path));
  }
}

void PD_SetProgFile(PD_AnalysisConfig* config, const char* x) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.SetProgFile(std::string(x));
}

void PD_SetParamsFile(PD_AnalysisConfig* config, const char* x) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.SetParamsFile(std::string(x));
}

void PD_SetOptimCacheDir(PD_AnalysisConfig* config, const char* opt_cache_dir) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.SetOptimCacheDir(std::string(opt_cache_dir));
}

const char* PD_ModelDir(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.model_dir().c_str();
}

const char* PD_ProgFile(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.prog_file().c_str();
}

const char* PD_ParamsFile(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.params_file().c_str();
}

void PD_EnableUseGpu(PD_AnalysisConfig* config, int memory_pool_init_size_mb,
                     int device_id) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.EnableUseGpu(static_cast<uint64_t>(memory_pool_init_size_mb),
                              device_id);
}

void PD_DisableGpu(PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.DisableGpu();
}

bool PD_UseGpu(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.use_gpu();
}

int PD_GpuDeviceId(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.gpu_device_id();
}

int PD_MemoryPoolInitSizeMb(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.memory_pool_init_size_mb();
}

float PD_FractionOfGpuMemoryForPool(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.fraction_of_gpu_memory_for_pool();
}

void PD_EnableCUDNN(PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.EnableCUDNN();
}

bool PD_CudnnEnabled(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.cudnn_enabled();
}

void PD_SwitchIrOptim(PD_AnalysisConfig* config, bool x) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.SwitchIrOptim(x);
}

bool PD_IrOptim(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.ir_optim();
}

void PD_SwitchUseFeedFetchOps(PD_AnalysisConfig* config, bool x) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.SwitchUseFeedFetchOps(x);
}

bool PD_UseFeedFetchOpsEnabled(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.use_feed_fetch_ops_enabled();
}

void PD_SwitchSpecifyInputNames(PD_AnalysisConfig* config, bool x) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.SwitchSpecifyInputNames(x);
}

bool PD_SpecifyInputName(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.specify_input_name();
}

void PD_EnableTensorRtEngine(PD_AnalysisConfig* config, int workspace_size,
                             int max_batch_size, int min_subgraph_size,
                             Precision precision, bool use_static,
                             bool use_calib_mode) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.EnableTensorRtEngine(
      workspace_size, max_batch_size, min_subgraph_size,
      paddle::ConvertToACPrecision(precision), use_static, use_calib_mode);
}

bool PD_TensorrtEngineEnabled(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.tensorrt_engine_enabled();
}

void PD_SwitchIrDebug(PD_AnalysisConfig* config, bool x) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.SwitchIrDebug(x);
}

void PD_EnableNgraph(PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.EnableNgraph();
}

bool PD_NgraphEnabled(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.ngraph_enabled();
}

void PD_EnableMKLDNN(PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.EnableMKLDNN();
}

void PD_SetMkldnnCacheCapacity(PD_AnalysisConfig* config, int capacity) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.SetMkldnnCacheCapacity(capacity);
}

bool PD_MkldnnEnabled(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.mkldnn_enabled();
}

void PD_SetCpuMathLibraryNumThreads(PD_AnalysisConfig* config,
                                    int cpu_math_library_num_threads) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.SetCpuMathLibraryNumThreads(cpu_math_library_num_threads);
}

int PD_CpuMathLibraryNumThreads(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.cpu_math_library_num_threads();
}

void PD_EnableMkldnnQuantizer(PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.EnableMkldnnQuantizer();
}

bool PD_MkldnnQuantizerEnabled(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.mkldnn_quantizer_enabled();
}

void PD_SetModelBuffer(PD_AnalysisConfig* config, const char* prog_buffer,
                       size_t prog_buffer_size, const char* params_buffer,
                       size_t params_buffer_size) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.SetModelBuffer(prog_buffer, prog_buffer_size, params_buffer,
                                params_buffer_size);
}

bool PD_ModelFromMemory(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.model_from_memory();
}

void PD_EnableMemoryOptim(PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.EnableMemoryOptim();
}

bool PD_MemoryOptimEnabled(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.enable_memory_optim();
}

void PD_EnableProfile(PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.EnableProfile();
}

bool PD_ProfileEnabled(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.profile_enabled();
}

void PD_SetInValid(PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  config->config.SetInValid();
}

bool PD_IsValid(const PD_AnalysisConfig* config) {
  PADDLE_ENFORCE_NOT_NULL(config);
  return config->config.is_valid();
}

void PD_DisableGlogInfo(PD_AnalysisConfig* config) {
  config->config.DisableGlogInfo();
}

void PD_DeletePass(PD_AnalysisConfig* config, char* pass_name) {
  return config->config.pass_builder()->DeletePass(std::string(pass_name));
}
}  // extern "C"
