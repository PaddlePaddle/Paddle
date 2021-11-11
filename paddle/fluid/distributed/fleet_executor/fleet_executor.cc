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

#include "paddle/fluid/distributed/fleet_executor/fleet_executor.h"
#include "paddle/fluid/distributed/fleet_executor/message_bus.h"
#include "paddle/fluid/distributed/fleet_executor/runtime_graph.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace distributed {

FleetExecutor::FleetExecutor(const std::string& exe_desc_str) {
  bool parse_flag = exe_desc_.ParseFromString(exe_desc_str);
  PADDLE_ENFORCE(parse_flag, platform::errors::PreconditionNotMet(
                                 "Error occurs while parsing string to proto"));
}

FleetExecutor::~FleetExecutor() {
  // Destroy Executor
}

void FleetExecutor::Init(const paddle::framework::ProgramDesc& program_desc) {
  // Compile and Initialize
  InitMessageBus();
}

void FleetExecutor::InitMessageBus() {
  std::stringstream ss;
  ss << "\nThe DNS table of the message bus is: \n";
  int64_t cur_rank = exe_desc_.cur_rank();
  std::unordered_map<int64_t, int64_t> interceptor_id_to_rank;
  std::unordered_map<int64_t, std::string> rank_to_addr;
  std::string addr;
  for (const auto& rank_info : exe_desc_.cluster_info()) {
    int64_t rank = rank_info.rank();
    std::string ip_port = rank_info.ip_port();
    ss << rank << "\t->\t" << ip_port << "\n";
    // TODO(Yuang): replace the first 'rank' with real interceptor id
    interceptor_id_to_rank.insert(std::make_pair(rank, rank));
    rank_to_addr.insert(std::make_pair(rank, ip_port));
    if (rank == cur_rank) {
      addr = ip_port;
    }
  }
  PADDLE_ENFORCE_NE(
      addr, "",
      platform::errors::NotFound(
          "Current rank is %s, which ip_port cannot be found in the config.",
          cur_rank));
  VLOG(3) << "Current rank is " << cur_rank << " and the ip_port is " << addr
          << ".";
  VLOG(3) << "The number of ranks are " << interceptor_id_to_rank.size() << ".";
  VLOG(5) << ss.str();
  MessageBus& message_bus_instance = MessageBus::Instance();
  if (!message_bus_instance.IsInit()) {
    message_bus_instance.Init(interceptor_id_to_rank, rank_to_addr, addr);
  }
}

void FleetExecutor::Run() {
  // Run
}

void FleetExecutor::Release() {
  // Release
}

std::shared_ptr<Carrier> FleetExecutor::GetCarrier() {
  // get carrier
  return nullptr;
}

}  // namespace distributed
}  // namespace paddle
