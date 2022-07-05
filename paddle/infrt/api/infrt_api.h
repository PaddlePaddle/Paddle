// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <memory>
#include <string>
#include <vector>

#include "paddle/phi/core/dense_tensor.h"

namespace infrt {

class InfRtConfig {
  std::string model_dir_;
  std::string param_dir_;
  std::vector<std::string> shared_libs_;

  // TODO(wilber): Design an easy-to-use interface.
  bool gpu_enabled_{false};
  bool tensorrt_enabled_{false};

 public:
  InfRtConfig() = default;
  void set_model_dir(const std::string& model_dir) { model_dir_ = model_dir; }
  const std::string& model_dir() const { return model_dir_; }

  void set_param_dir(const std::string& param_dir) { param_dir_ = param_dir; }
  const std::string& param_dir() const { return param_dir_; }

  void set_shared_libs(const std::vector<std::string>& shared_libs) {
    shared_libs_ = shared_libs;
  }
  const std::vector<std::string>& shared_libs() const { return shared_libs_; }

  void enable_gpu() { gpu_enabled_ = true; }
  bool gpu_enabled() const { return gpu_enabled_; }

  // TODO(wilber): Design an easy-to-use interface.
  void enable_tensorrt() { tensorrt_enabled_ = true; }
  void disable_tensorrt() { tensorrt_enabled_ = false; }
  bool tensorrt_enabled() const { return tensorrt_enabled_; }

  virtual ~InfRtConfig() = default;
};

class InfRtPredictor {
 public:
  InfRtPredictor();
  ~InfRtPredictor();
  void Run();
  int Init(const InfRtConfig& config);
  int GetInputNum();
  ::phi::DenseTensor* GetInput(int i);
  int GetOutputNum();
  ::phi::DenseTensor* GetOutput(int i);

 protected:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::unique_ptr<InfRtPredictor> CreateInfRtPredictor(const InfRtConfig& config);

}  // namespace infrt
