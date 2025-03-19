/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <string>
#include <vector>

#if defined(PADDLE_WITH_CUDA)
#include <cuda_runtime.h>
#endif

#include "paddle/common/flags.h"
#include "paddle/fluid/inference/capi_exp/pd_inference_api.h"

PD_DEFINE_string(infer_model, "", "model path");

namespace paddle {
namespace inference {
namespace analysis {

TEST(PD_Config, gpu_interface) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  std::string prog_file = model_dir + "/__model__";
  std::string param_file = model_dir + "/__params__";
  std::string opt_cache_dir = FLAGS_infer_model + "/OptimCacheDir";
  const char* ops_name = "conv_2d";

  PD_Config* config = PD_ConfigCreate();
  PD_ConfigSetModel(config, prog_file.c_str(), param_file.c_str());
  PD_ConfigSetOptimCacheDir(config, opt_cache_dir.c_str());

  PD_ConfigEnableUseGpu(config, 100, 0, 0);
  bool use_gpu = PD_ConfigUseGpu(config);
  EXPECT_TRUE(use_gpu);
  int init_size = PD_ConfigMemoryPoolInitSizeMb(config);
  EXPECT_EQ(init_size, 100);
  int gpu_device_id = PD_ConfigGpuDeviceId(config);
  EXPECT_EQ(gpu_device_id, 0);
  float frac = PD_ConfigFractionOfGpuMemoryForPool(config);
  LOG(INFO) << frac;
  PD_ConfigEnableCudnn(config);
  bool cudnn = PD_ConfigCudnnEnabled(config);
  EXPECT_TRUE(cudnn);

  PD_ConfigEnableTensorRtEngine(
      config, 1 << 20, 1, 3, PD_PRECISION_INT8, FALSE, TRUE);
  bool trt_enable = PD_ConfigTensorRtEngineEnabled(config);
  EXPECT_TRUE(trt_enable);

  const char* tensor_name = "image";
  std::array<size_t, 1> shapes_num = {4};
  std::array<int32_t, 4> min_shape = {1, 3, 36, 36};
  std::array<int32_t, 4> max_shape = {1, 3, 224, 224};
  std::array<int32_t, 4> opt_shape = {1, 3, 224, 224};
  int32_t* min_shape_ptr = min_shape.data();
  int32_t* max_shape_ptr = max_shape.data();
  int32_t* opt_shape_ptr = opt_shape.data();
  PD_ConfigSetTrtDynamicShapeInfo(config,
                                  1,
                                  &tensor_name,
                                  shapes_num.data(),
                                  &min_shape_ptr,
                                  &max_shape_ptr,
                                  &opt_shape_ptr,
                                  FALSE);
  PD_ConfigDisableTensorRtOPs(config, 1, &ops_name);
  PD_ConfigEnableVarseqlen(config);
  bool oss_enabled = PD_ConfigTensorRtOssEnabled(config);
  EXPECT_TRUE(oss_enabled);

  PD_ConfigEnableTensorRtDla(config, 4);
  bool dla_enabled = PD_ConfigTensorRtDlaEnabled(config);
  EXPECT_TRUE(dla_enabled);

  PD_ConfigEnableGpuMultiStream(config);
  bool thread_local_thread = PD_ConfigThreadLocalStreamEnabled(config);
  EXPECT_TRUE(thread_local_thread);

#if defined(PADDLE_WITH_CUDA)
  {
    cudaStream_t external_stream;
    cudaStreamCreate(&external_stream);
    PD_ConfigSetExecStream(config, external_stream);
  }
#endif

  PD_ConfigDisableGpu(config);
  PD_ConfigDestroy(config);
}

TEST(PD_Config, use_gpu) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  PD_Config* config = PD_ConfigCreate();

  PD_ConfigDisableGpu(config);
  PD_ConfigSetCpuMathLibraryNumThreads(config, 10);
  int num_thread = PD_ConfigGetCpuMathLibraryNumThreads(config);
  EXPECT_EQ(num_thread, 10);

  PD_ConfigSwitchIrDebug(config, TRUE);
  PD_ConfigSetModelDir(config, model_dir.c_str());
  PD_ConfigSetOptimCacheDir(config,
                            (FLAGS_infer_model + "/OptimCacheDir").c_str());
  const char* model_dir_ = PD_ConfigGetModelDir(config);
  LOG(INFO) << model_dir_;

  PD_ConfigEnableUseGpu(config, 100, 0, 0);
  bool use_gpu = PD_ConfigUseGpu(config);
  EXPECT_TRUE(use_gpu);
  int device_id = PD_ConfigGpuDeviceId(config);
  EXPECT_EQ(device_id, 0);
  int init_size = PD_ConfigMemoryPoolInitSizeMb(config);
  EXPECT_EQ(init_size, 100);

  float frac = PD_ConfigFractionOfGpuMemoryForPool(config);
  LOG(INFO) << frac;

  PD_ConfigEnableCudnn(config);
  bool cudnn = PD_ConfigCudnnEnabled(config);
  EXPECT_TRUE(cudnn);

  PD_ConfigSwitchIrOptim(config, TRUE);
  bool ir_optim = PD_ConfigIrOptim(config);
  EXPECT_TRUE(ir_optim);

  PD_ConfigEnableTensorRtEngine(
      config, 1 << 20, 1, 3, PD_PRECISION_FLOAT32, FALSE, FALSE);
  bool trt_enable = PD_ConfigTensorRtEngineEnabled(config);
  EXPECT_TRUE(trt_enable);
  PD_ConfigEnableMemoryOptim(config, true);
  bool memory_optim_enable = PD_ConfigMemoryOptimEnabled(config);
  EXPECT_TRUE(memory_optim_enable);
  PD_ConfigEnableProfile(config);
  bool profiler_enable = PD_ConfigProfileEnabled(config);
  EXPECT_TRUE(profiler_enable);
  PD_ConfigSetInvalid(config);
  bool is_valid = PD_ConfigIsValid(config);
  EXPECT_FALSE(is_valid);
  PD_ConfigDestroy(config);
}

TEST(PD_Config, trt_int8) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  PD_Config* config = PD_ConfigCreate();
  PD_ConfigEnableUseGpu(config, 100, 0, 0);
  PD_ConfigEnableTensorRtEngine(
      config, 1 << 20, 1, 3, PD_PRECISION_INT8, FALSE, TRUE);
  bool trt_enable = PD_ConfigTensorRtEngineEnabled(config);
  EXPECT_TRUE(trt_enable);
  PD_ConfigDestroy(config);
}

TEST(PD_Config, trt_fp16) {
  std::string model_dir = FLAGS_infer_model + "/mobilenet";
  PD_Config* config = PD_ConfigCreate();
  PD_ConfigEnableUseGpu(config, 100, 0, 0);
  PD_ConfigEnableTensorRtEngine(
      config, 1 << 20, 1, 3, PD_PRECISION_HALF, FALSE, FALSE);
  bool trt_enable = PD_ConfigTensorRtEngineEnabled(config);
  EXPECT_TRUE(trt_enable);
  PD_Predictor* predictor = PD_PredictorCreate(config);
  PD_PredictorDestroy(predictor);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
