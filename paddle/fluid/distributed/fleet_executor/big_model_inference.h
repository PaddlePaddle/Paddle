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
#include <string>
#include <vector>
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {
class ProgramDesc;
class Scope;
class Tensor;
}

namespace distributed {

class BigModelInfConfig {
 public:
  BigModelInfConfig() = default;
  ~BigModelInfConfig() = default;

 private:
  std::string model_dir_;
  std::vector<std::string> trainer_endpoints_{};
  std::string current_endpoints_{};
  int64_t nranks_;
  int64_t local_rank_;
  int64_t device_id_;
  int64_t mp_degree_;
  int64_t pp_degree_;
};

class BigModelInf {
 public:
  explicit BigModelInf(const BigModelInfConfig& config) : config_(config) {}
  void Init() { /* TODO(fleet exe dev): implement this funct */
  }
  void Run(const std::vector<framework::Tensor>& input_data,
           std::vector<framework::Tensor>* output_data) {
    /* TODO(fleet exe dev): implement this funct */
  }
  ~BigModelInf() = default;

 private:
  DISABLE_COPY_AND_ASSIGN(BigModelInf);
  BigModelInfConfig config_;
};

}  // namespace distributed
}  // namespace paddle
