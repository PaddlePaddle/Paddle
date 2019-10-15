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

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#if defined(_WIN32)
#ifdef PADDLE_ON_INFERENCE
#define PADDLE_CAPI_EXPORT __declspec(dllexport)
#else
#define PADDLE_CAPI_EXPORT __declspec(dllimport)
#endif  // PADDLE_ON_INFERENCE
#else
#define PADDLE_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

#ifdef __cplusplus
extern "C" {
#endif

enum PD_DataType { PD_FLOAT32, PD_INT32, PD_INT64, PD_UINT8, PD_UNKDTYPE };
typedef struct PD_PaddleBuf PD_PaddleBuf;
typedef struct PD_AnalysisConfig PD_AnalysisConfig;

typedef struct PD_ZeroCopyData {
  char* name;
  void* data;
  PD_DataType dtype;
  int* shape;
  int shape_size;
} PD_ZeroCopyData;
typedef struct InTensorShape {
  char* name;
  int* tensor_shape;
  int shape_size;
} InTensorShape;

PADDLE_CAPI_EXPORT extern PD_PaddleBuf* PD_NewPaddleBuf();

PADDLE_CAPI_EXPORT extern void PD_DeletePaddleBuf(PD_PaddleBuf* buf);

PADDLE_CAPI_EXPORT extern void PD_PaddleBufResize(PD_PaddleBuf* buf,
                                                  size_t length);

PADDLE_CAPI_EXPORT extern void PD_PaddleBufReset(PD_PaddleBuf* buf, void* data,
                                                 size_t length);

PADDLE_CAPI_EXPORT extern bool PD_PaddleBufEmpty(PD_PaddleBuf* buf);

PADDLE_CAPI_EXPORT extern void* PD_PaddleBufData(PD_PaddleBuf* buf);

PADDLE_CAPI_EXPORT extern size_t PD_PaddleBufLength(PD_PaddleBuf* buf);

// PaddleTensor
typedef struct PD_Tensor PD_Tensor;

PADDLE_CAPI_EXPORT extern PD_Tensor* PD_NewPaddleTensor();

PADDLE_CAPI_EXPORT extern void PD_DeletePaddleTensor(PD_Tensor* tensor);

PADDLE_CAPI_EXPORT extern void PD_SetPaddleTensorName(PD_Tensor* tensor,
                                                      char* name);

PADDLE_CAPI_EXPORT extern void PD_SetPaddleTensorDType(PD_Tensor* tensor,
                                                       PD_DataType dtype);

PADDLE_CAPI_EXPORT extern void PD_SetPaddleTensorData(PD_Tensor* tensor,
                                                      PD_PaddleBuf* buf);

PADDLE_CAPI_EXPORT extern void PD_SetPaddleTensorShape(PD_Tensor* tensor,
                                                       int* shape, int size);

PADDLE_CAPI_EXPORT extern const char* PD_GetPaddleTensorName(
    const PD_Tensor* tensor);

PADDLE_CAPI_EXPORT extern PD_DataType PD_GetPaddleTensorDType(
    const PD_Tensor* tensor);

PADDLE_CAPI_EXPORT extern PD_PaddleBuf* PD_GetPaddleTensorData(
    const PD_Tensor* tensor);

PADDLE_CAPI_EXPORT extern int* PD_GetPaddleTensorShape(const PD_Tensor* tensor,
                                                       int** size);

// AnalysisPredictor
PADDLE_CAPI_EXPORT extern bool PD_PredictorRun(const PD_AnalysisConfig* config,
                                               PD_Tensor* inputs, int in_size,
                                               PD_Tensor* output_data,
                                               int** out_size, int batch_size);

PADDLE_CAPI_EXPORT extern bool PD_PredictorZeroCopyRun(
    const PD_AnalysisConfig* config, PD_ZeroCopyData* inputs, int in_size,
    PD_ZeroCopyData** output, int** out_size);

// AnalysisConfig
enum Precision { kFloat32 = 0, kInt8, kHalf };

PADDLE_CAPI_EXPORT extern PD_AnalysisConfig* PD_NewAnalysisConfig();

PADDLE_CAPI_EXPORT extern void PD_DeleteAnalysisConfig(
    PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern void PD_SetModel(PD_AnalysisConfig* config,
                                           const char* model_dir,
                                           const char* params_path = NULL);

PADDLE_CAPI_EXPORT
extern void PD_SetProgFile(PD_AnalysisConfig* config, const char* x);

PADDLE_CAPI_EXPORT extern void PD_SetParamsFile(PD_AnalysisConfig* config,
                                                const char* x);

PADDLE_CAPI_EXPORT extern void PD_SetOptimCacheDir(PD_AnalysisConfig* config,
                                                   const char* opt_cache_dir);

PADDLE_CAPI_EXPORT extern const char* PD_ModelDir(
    const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern const char* PD_ProgFile(
    const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern const char* PD_ParamsFile(
    const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern void PD_EnableUseGpu(
    PD_AnalysisConfig* config, uint64_t memory_pool_init_size_mb,
    int device_id = 0);

PADDLE_CAPI_EXPORT extern void PD_DisableGpu(PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern bool PD_UseGpu(const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern int PD_GpuDeviceId(const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern int PD_MemoryPoolInitSizeMb(
    const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern float PD_FractionOfGpuMemoryForPool(
    const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern void PD_EnableCUDNN(PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern bool PD_CudnnEnabled(const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern void PD_SwitchIrOptim(PD_AnalysisConfig* config,
                                                bool x = true);

PADDLE_CAPI_EXPORT extern bool PD_IrOptim(const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern void PD_SwitchUseFeedFetchOps(
    PD_AnalysisConfig* config, bool x = true);

PADDLE_CAPI_EXPORT extern bool PD_UseFeedFetchOpsEnabled(
    const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern void PD_SwitchSpecifyInputNames(
    PD_AnalysisConfig* config, bool x = true);

PADDLE_CAPI_EXPORT extern bool PD_SpecifyInputName(
    const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern void PD_EnableTensorRtEngine(
    PD_AnalysisConfig* config, int workspace_size = 1 << 20,
    int max_batch_size = 1, int min_subgraph_size = 3,
    Precision precision = Precision::kFloat32, bool use_static = false,
    bool use_calib_mode = false);

PADDLE_CAPI_EXPORT extern bool PD_TensorrtEngineEnabled(
    const PD_AnalysisConfig* config);

typedef struct PD_MaxInputShape {
  char* name;
  int* shape;
  int shape_size;
} PD_MaxInputShape;

PADDLE_CAPI_EXPORT extern void PD_EnableAnakinEngine(
    PD_AnalysisConfig* config, int max_batch_size = 1,
    PD_MaxInputShape* max_input_shape = NULL, int max_input_shape_size = 0,
    int min_subgraph_size = 6, Precision precision = Precision::kFloat32,
    bool auto_config_layout = false, char** passes_filter = NULL,
    int passes_filter_size = 0, char** ops_filter = NULL,
    int ops_filter_size = 0);

PADDLE_CAPI_EXPORT extern bool PD_AnakinEngineEnabled(
    const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern void PD_SwitchIrDebug(PD_AnalysisConfig* config,
                                                bool x = true);

PADDLE_CAPI_EXPORT extern void PD_EnableNgraph(PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern bool PD_NgraphEnabled(
    const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern void PD_EnableMKLDNN(PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern void PD_SetMkldnnCacheCapacity(
    PD_AnalysisConfig* config, int capacity);

PADDLE_CAPI_EXPORT extern bool PD_MkldnnEnabled(
    const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern void PD_SetCpuMathLibraryNumThreads(
    PD_AnalysisConfig* config, int cpu_math_library_num_threads);

PADDLE_CAPI_EXPORT extern int PD_CpuMathLibraryNumThreads(
    const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern void PD_EnableMkldnnQuantizer(
    PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern bool PD_MkldnnQuantizerEnabled(
    const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern void PD_SetModelBuffer(PD_AnalysisConfig* config,
                                                 const char* prog_buffer,
                                                 size_t prog_buffer_size,
                                                 const char* params_buffer,
                                                 size_t params_buffer_size);

PADDLE_CAPI_EXPORT extern bool PD_ModelFromMemory(
    const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern void PD_EnableMemoryOptim(PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern bool PD_MemoryOptimEnabled(
    const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern void PD_EnableProfile(PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern bool PD_ProfileEnabled(
    const PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern void PD_SetInValid(PD_AnalysisConfig* config);

PADDLE_CAPI_EXPORT extern bool PD_IsValid(const PD_AnalysisConfig* config);

#ifdef __cplusplus
}  // extern "C"
#endif
