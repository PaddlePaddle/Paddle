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

#include "paddle/fluid/distributed/fleet_executor/fleet_executor_desc.pb.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
class ProgramDesc;
class Scope;
class Tensor;
}

namespace distributed {

struct DistModelConfig {
  std::string model_dir{};
  std::vector<std::string> trainer_endpoints{};
  std::string current_endpoint{};
  int64_t nranks{1};
  int64_t local_rank{0};
  int64_t device_id{0};
  int64_t mp_degree{1};
  int64_t pp_degree{1};
};

class DistModel {
 public:
  explicit DistModel(const DistModelConfig& config) : config_(config) {}
  bool Init();
  void Run(const std::vector<framework::Tensor>& input_data,
           std::vector<framework::Tensor>* output_data);
  ~DistModel() = default;

 private:
  DISABLE_COPY_AND_ASSIGN(DistModel);

  bool PrepareScope();
  bool PrepareProgram();
  bool LoadProgram();
  bool LoadParameters();
  bool CommInit();

  DistModelConfig config_;
  FleetExecutorDesc executor_desc_;
  platform::Place place_;
  std::shared_ptr<framework::Scope> scope_;
  std::shared_ptr<framework::ProgramDesc> program_;
};

}  // namespace distributed
}  // namespace paddle
