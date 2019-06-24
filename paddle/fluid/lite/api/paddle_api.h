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

/*
 * This file defines PaddlePredictor, the api for lite. It supports multiple
 * hardware including ARM, X86, OpenCL, CUDA and so on.
 */

#ifndef PADDLE_LITE_API_H_  // NOLINT
#define PADDLE_LITE_API_H_
#include <memory>
#include <string>
#include <vector>
#include "paddle_place.h"  // NOLINT

namespace paddle {
namespace lite_api {

using shape_t = std::vector<int64_t>;

struct Tensor {
  explicit Tensor(void* raw);
  explicit Tensor(const void* raw);

  void Resize(const shape_t& shape);

  /// Readonly data.
  template <typename T>
  const T* data() const;

  template <typename T>
  T* mutable_data() const;

  /// Shape of the tensor.
  shape_t shape() const;

 private:
  void* raw_tensor_;
};

/// The PaddlePredictor defines the basic interfaces for different kinds of
/// predictors.
class PaddlePredictor {
 public:
  PaddlePredictor() = default;

  /// Get i-th input.
  virtual std::unique_ptr<Tensor> GetInput(int i) = 0;

  /// Get i-th output.
  virtual std::unique_ptr<const Tensor> GetOutput(int i) const = 0;

  virtual void Run() = 0;

  /// Get a readonly tensor, return null if no one called `name` exists.
  virtual std::unique_ptr<const Tensor> GetTensor(
      const std::string& name) const = 0;

  /// Persist the optimized model to disk. This API is only supported by
  /// CxxConfig, and the persisted model can be reused for MobileConfig.
  virtual void SaveOptimizedModel(const std::string& model_dir);

  virtual ~PaddlePredictor() = default;
};

/// Base class for all the configs.
class ConfigBase {
  std::string model_dir_;

 public:
  void set_model_dir(const std::string& x) { model_dir_ = x; }

  const std::string& model_dir() const { return model_dir_; }
};

/// CxxConfig is the config for the Full feature predictor.
class CxxConfig : public ConfigBase {
  Place preferred_place_;
  std::vector<Place> valid_places_;

 public:
  void set_preferred_place(const Place& x) { preferred_place_ = x; }
  void set_valid_places(const std::vector<Place>& x) { valid_places_ = x; }

  const Place& preferred_place() const { return preferred_place_; }
  const std::vector<Place>& valid_places() const { return valid_places_; }
};

/// MobileConfig is the config for the light weight predictor, it will skip
/// IR optimization or other unnecessary stages.
class MobileConfig : public ConfigBase {};

template <typename ConfigT>
std::shared_ptr<PaddlePredictor> CreatePaddlePredictor(const ConfigT&);

}  // namespace lite_api
}  // namespace paddle

#endif  // NOLINT
