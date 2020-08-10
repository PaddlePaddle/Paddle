/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*
 * This file contains the definition of a simple Inference API for Paddle.
 *
 * ATTENTION: It requires some C++11 features, for lower version C++ or C, we
 * might release another API.
 */

#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle_analysis_config.h"  // NOLINT
#include "paddle_api.h"              // NOLINT

namespace paddle_infer {
enum PaddleDType {
  FLOAT32,
  INT64,
  INT32,
  UINT8,
};

enum class PaddlePlace {
  kUNK = -1,
  kCPU,
  kGPU,
};

struct PD_INFER_DECL Config {
  Config() = default;
  enum class Precision {
    kFloat32 = 0,  ///< fp32
    kInt8,         ///< int8
    kHalf,         ///< fp16
  };
  // Must be configured
  explicit Config(const Config& config) = delete;
  explicit Config(const std::string& model_dir);
  explicit Config(const std::string& prog_file, const std::string& params_file);
  void SetModel(const std::string& model_dir);
  void SetModel(const std::string& prog_file, const std::string& params_file);
  void SetModelBuffer(const char* prog_buffer, size_t prog_buffer_size,
                      const char* params_buffer, size_t params_buffer_size);
  // Optional configuration
  void EnableMemoryOptim();

  // cpu
  void EnableMKLDNN();
  void SetMkldnnCacheCapacity(int capacity);

  void SetCpuMathLibraryNumThreads(int cpu_math_library_num_threads);
  void EnableMkldnnQuantizer();
  paddle::MkldnnQuantizerConfig* mkldnn_quantizer_config() const;

  // gpu
  void EnableUseGpu(uint64_t memory_pool_init_size_mb, int device_id = 0);
  void EnableTensorRtEngine(int workspace_size = 1 << 20,
                            int max_batch_size = 1, int min_subgraph_size = 3,
                            Precision precision = Precision::kFloat32,
                            bool use_static = false,
                            bool use_calib_mode = true);
  void SetTRTDynamicShapeInfo(
      std::map<std::string, std::vector<int>> min_input_shape,
      std::map<std::string, std::vector<int>> max_input_shape,
      std::map<std::string, std::vector<int>> optim_input_shape,
      bool disable_trt_plugin_fp16 = false);

  void EnableGpuMultiStream();

  // xpu
  void EnableLiteEngine(Precision precision = Precision::kFloat32,
                        bool zero_copy = false,
                        const std::vector<std::string>& passes_filter = {},
                        const std::vector<std::string>& ops_filter = {});
  void EnableXpu(int l3_workspace_size = 0xfffc00);

  // debug
  void SwitchIrOptim(int x = true);
  void EnableProfile();
  void DisableGlogInfo();
  paddle::PassStrategy* pass_builder() const;

  // not for users
  void SetInValid() const { config_.SetInValid(); }
  void PartiallyRelease() { config_.PartiallyRelease(); }
  paddle::AnalysisConfig& get_analysis_config() { return config_; }

 private:
  void Init() { config_.SwitchUseFeedFetchOps(false); }
  paddle::AnalysisConfig::Precision PrecisionTrait(
      Config::Precision precision) {
    switch (precision) {
      case Config::Precision::kFloat32:
        return paddle::AnalysisConfig::Precision::kFloat32;
      case Config::Precision::kInt8:
        return paddle::AnalysisConfig::Precision::kInt8;
      case Config::Precision::kHalf:
        return paddle::AnalysisConfig::Precision::kHalf;
      default:
        throw std::runtime_error(
            "Wrong Precision: Your precion specific should be kFloat32, kInt8, "
            "kHalf");
    }
  }
  paddle::AnalysisConfig config_;
};

class PD_INFER_DECL Tensor {
 public:
  // can only be created by predictor->GetInputHandle(cosnt std::string& name)
  // or predictor->GetOutputHandle(cosnt std::string& name)
  Tensor() = delete;
  explicit Tensor(std::unique_ptr<paddle::ZeroCopyTensor> tensor)
      : tensor_(std::move(tensor)) {}
  void Reshape(const std::vector<int>& shape);

  template <typename T>
  void CopyFromCpu(const T* data);

  // should add the place
  template <typename T>
  T* mutable_data(PaddlePlace place);

  template <typename T>
  void CopyToCpu(T* data);

  template <typename T>
  T* data(PaddlePlace* place, int* size) const;

  void SetLoD(const std::vector<std::vector<size_t>>& x);
  std::vector<std::vector<size_t>> lod() const;

  void SetPlace(PaddlePlace place, int device = -1);
  PaddleDType type() const;

  std::vector<int> shape() const;
  const std::string& name() const;

 private:
  std::unique_ptr<paddle::ZeroCopyTensor> tensor_;
};

class PD_INFER_DECL Predictor {
 public:
  Predictor() = delete;
  ~Predictor() {}
  explicit Predictor(std::unique_ptr<paddle::PaddlePredictor> pred)
      : predictor_(std::move(pred)) {}

  explicit Predictor(Config& config) {
    predictor_ = paddle::CreatePaddlePredictor(config.get_analysis_config());
  }

  std::vector<std::string> GetInputNames();
  std::unique_ptr<Tensor> GetInputHandle(const std::string& name);

  bool Run();

  std::vector<std::string> GetOutputNames();
  std::unique_ptr<Tensor> GetOutputHandle(const std::string& name);

  std::shared_ptr<Predictor> Clone();
  void ClearIntermediateTensor();

 private:
  std::unique_ptr<paddle::PaddlePredictor> predictor_;
};

PD_INFER_DECL std::shared_ptr<Predictor> CreatePredictor(
    Config& config);  // NOLINT
PD_INFER_DECL int PaddleDtypeSize(PaddleDType dtype);
PD_INFER_DECL std::shared_ptr<paddle::framework::Cipher> MakeCipher(
    const std::string& config_file);

extern "C" {
PD_INFER_DECL std::string GetPaddleVersion();
PD_INFER_DECL std::string UpdateDllFlag(const char* name, const char* value);
}

}  // namespace paddle_infer
