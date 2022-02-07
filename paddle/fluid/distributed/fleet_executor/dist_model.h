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

#include "paddle/fluid/distributed/fleet_executor/dist_model_tensor_wrapper.h"
#include "paddle/fluid/distributed/fleet_executor/fleet_executor_desc.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {

namespace framework {
class ProgramDesc;
class Scope;
class BlockDesc;
}

namespace distributed {

class TaskNode;
class FleetExecutor;

struct DistModelConfig {
  std::string model_dir{};
  framework::ProgramDesc* program_desc{nullptr};
  framework::Scope* scope{nullptr};
  std::string place{};
  int64_t device_id{0};
  std::vector<std::string> trainer_endpoints{};
  std::string current_endpoint{};
  int64_t nranks{1};
  int64_t local_rank{0};
  int64_t mp_degree{1};
  int64_t pp_degree{1};
  int64_t mp_ring_id{-1};
  int64_t pp_upstream_ring_id{-1};
  int64_t pp_downstream_ring_id{-1};
  bool enable_timer{false};
};

class DistModel {
 public:
  explicit DistModel(const DistModelConfig& config) : config_(config) {}
  bool Init();
  bool Run(const std::vector<DistModelTensor>& input_data,
           std::vector<DistModelTensor>* output_data);
  ~DistModel() = default;

 private:
  DISABLE_COPY_AND_ASSIGN(DistModel);

  bool PrepareScope();
  bool PrepareProgram();
  bool LoadProgram();
  bool LoadParameters();
  bool PreparePlace();
  bool CommInit();
  bool PrepareFeedAndFetch();
  bool PrepareFleetExe();
  void InsertCommOp(std::string tmp_var_name, int nranks, int rank,
                    const std::vector<std::string>& peer_endpoints,
                    framework::BlockDesc* block, int ring_id);
  bool FeedData(const std::vector<DistModelTensor>& input_data,
                framework::Scope* scope);
  bool FetchResults(std::vector<DistModelTensor>* output_data,
                    framework::Scope* scope);
  template <typename T>
  bool FetchResult(const framework::LoDTensor& fetch,
                   DistModelTensor* output_data);

  std::string carrier_id_;
  std::vector<framework::LoDTensor> feed_tensors_;
  std::vector<framework::OpDesc*> feeds_;
  std::map<std::string, int64_t> feed_names_;
  std::map<int64_t, std::string> idx_to_feeds_;
  std::map<std::string, DistModelDataType> feeds_to_dtype_;
  std::vector<framework::OpDesc*> fetches_;
  std::map<int64_t, std::string> idx_to_fetches_;
  DistModelConfig config_;
  FleetExecutorDesc executor_desc_;
  std::shared_ptr<FleetExecutor> fleet_exe;
  std::shared_ptr<TaskNode> task_node_;
  std::shared_ptr<framework::Scope> scope_;
  paddle::platform::Place place_;
  std::shared_ptr<framework::ProgramDesc> program_;
};

}  // namespace distributed
}  // namespace paddle
