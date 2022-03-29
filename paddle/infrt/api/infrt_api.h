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

#include "paddle/infrt/tensor/dense_host_tensor.h"

namespace infrt {

class InfRtConfig {
  std::string model_dir_;
  std::string mlir_path_;
  std::vector<std::string> shared_libs_;

 public:
  InfRtConfig() = default;
  void set_model_dir(const std::string& model_dir) { model_dir_ = model_dir; }
  const std::string& model_dir() const { return model_dir_; }

  void set_mlir_path(const std::string& mlir_path) { mlir_path_ = mlir_path; }
  const std::string& mlir_path() const { return mlir_path_; }

  void set_shared_libs(const std::vector<std::string>& shared_libs) {
    shared_libs_ = shared_libs;
  }
  const std::vector<std::string>& shared_libs() const { return shared_libs_; }

  virtual ~InfRtConfig() = default;
};

class InfRtPredictor {
 public:
  InfRtPredictor();
  ~InfRtPredictor();
  void Run();
  int Init(const InfRtConfig& config);
  int GetInputNum();
  tensor::DenseHostTensor* GetInput(int i);
  int GetOutputNum();
  tensor::DenseHostTensor* GetOutput(int i);

 protected:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::shared_ptr<InfRtPredictor> CreateInfRtPredictor(const InfRtConfig& config);

}  // namespace infrt
