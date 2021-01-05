// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <gflags/gflags.h>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/generator.h"

namespace paddle {
namespace distributed {

class Initializer {
 public:
  Initializer() {}

  explicit Initializer(const std::vector<std::string> &attrs) {}

  virtual float GetValue() = 0;

  virtual ~Initializer() {}

 protected:
  std::string name_;
  unsigned int seed_;
};

class UniformInitializer : public Initializer {
 public:
  explicit UniformInitializer(const std::vector<std::string> &attrs) {
    name_ = attrs[0];
    seed_ = static_cast<unsigned int>(std::stoi(attrs[1]));
    min_ = std::stof(attrs[2]);
    max_ = std::stof(attrs[3]);

    dist_ = std::uniform_real_distribution<float>(min_, max_);
    random_engine_ = framework::GetCPURandomEngine(seed_);
  }

  float GetValue() override { return dist_(*random_engine_); }

 private:
  float min_;
  float max_;

  std::shared_ptr<std::mt19937_64> random_engine_;
  std::uniform_real_distribution<float> dist_;
};

class GaussianInitializer : public Initializer {
 public:
  explicit GaussianInitializer(const std::vector<std::string> &attrs) {
    name_ = attrs[0];
    seed_ = static_cast<unsigned int>(std::stoi(attrs[1]));
    mean_ = std::stof(attrs[2]);
    std_ = std::stof(attrs[3]);

    random_engine_ = framework::GetCPURandomEngine(seed_);

    dist_ = std::normal_distribution<float>(mean_, std_);
  }

  float GetValue() override { return dist_(*random_engine_); }

 private:
  float std_;
  float mean_;

  std::shared_ptr<std::mt19937_64> random_engine_;
  std::normal_distribution<float> dist_;
};

class FillConstantInitializer : public Initializer {
 public:
  explicit FillConstantInitializer(const std::vector<std::string> &attrs) {
    name_ = attrs[0];
    value_ = std::stof(attrs[1]);
  }

  float GetValue() override { return value_; }

 private:
  float value_;
};
}  // namespace distributed
}  // namespace paddle
