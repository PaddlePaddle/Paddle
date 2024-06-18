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

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "paddle/common/flags.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/kernels/funcs/truncated_normal.h"

namespace paddle {
namespace distributed {

class Initializer {
 public:
  Initializer() {}

  explicit Initializer(const std::vector<std::string> &attrs) {}

  virtual float GetValue() = 0;

  virtual void GetValue(std::vector<float> *values, int numel) {
    for (int x = 0; x < numel; ++x) {
      values->push_back(GetValue());
    }
  }

  virtual void GetValue(float *value, int numel) {
    for (int x = 0; x < numel; ++x) {
      value[x] = GetValue();
    }
  }

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
    random_engine_ = phi::GetCPURandomEngine(seed_);
  }

  float GetValue() override { return dist_(*random_engine_); }
  void GetValue(float *value, int numel) {
    for (int x = 0; x < numel; ++x) {
      value[x] = dist_(*random_engine_);
    }
  }

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

    random_engine_ = phi::GetCPURandomEngine(seed_);

    dist_ = std::normal_distribution<float>(mean_, std_);
  }

  float GetValue() override { return dist_(*random_engine_); }
  void GetValue(float *value, int numel) {
    for (int x = 0; x < numel; ++x) {
      value[x] = dist_(*random_engine_);
    }
  }

 private:
  float std_;
  float mean_;

  std::shared_ptr<std::mt19937_64> random_engine_;
  std::normal_distribution<float> dist_;
};

class TruncatedGaussianInitializer : public Initializer {
 public:
  explicit TruncatedGaussianInitializer(const std::vector<std::string> &attrs) {
    name_ = attrs[0];
    seed_ = static_cast<unsigned int>(std::stoi(attrs[1]));
    mean_ = std::stof(attrs[2]);
    std_ = std::stof(attrs[3]);
    a_ = std::stof(attrs[4]);
    b_ = std::stof(attrs[5]);

    std::uniform_real_distribution<float> dist_(
        std::numeric_limits<float>::min(), 1.0);
    random_engine_ = phi::GetCPURandomEngine(seed_);
  }

  float GetValue() override {
    TruncatedNormal<float> truncated_normal(mean_, std_, a_, b_);
    float value = truncated_normal(dist_(*random_engine_));
    return value;
  }

  void GetValue(float *value, int numel) {
    TruncatedNormal<float> truncated_normal(mean_, std_, a_, b_);
    for (int x = 0; x < numel; ++x) {
      value[x] = truncated_normal(dist_(*random_engine_));
    }
  }

 private:
  float std_;
  float mean_;
  float a_;
  float b_;

  std::shared_ptr<std::mt19937_64> random_engine_;
  std::uniform_real_distribution<float> dist_;
};

class FillConstantInitializer : public Initializer {
 public:
  explicit FillConstantInitializer(const std::vector<std::string> &attrs) {
    name_ = attrs[0];
    value_ = std::stof(attrs[1]);
  }

  float GetValue() override { return value_; }
  void GetValue(float *value, int numel) { std::fill_n(value, numel, value_); }

 private:
  float value_;
};
}  // namespace distributed
}  // namespace paddle
