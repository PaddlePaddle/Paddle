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
#include "paddle/fluid/distributed/fleet_executor/carrier.h"
#include "paddle/fluid/distributed/fleet_executor/message_bus.h"
#include "paddle/fluid/distributed/fleet_executor/runtime_graph.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/operator.h"
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
  runtime_graph_ = std::make_unique<RuntimeGraph>(program_desc, exe_desc_);
  InitCarrier();
  InitMessageBus();
}

void FleetExecutor::InitCarrier() {
  Carrier& carrier_instance = Carrier::Instance();
  if (!carrier_instance.IsInit()) {
    carrier_instance.Init(runtime_graph_->intercepter_id_to_node());
  }
}

void FleetExecutor::InitMessageBus() {
  std::stringstream ss;
  ss << "\nThe DNS table of the message bus is: \n";
  int64_t cur_rank = exe_desc_.cur_rank();
  std::unordered_map<int64_t, std::string> rank_to_addr;
  std::string addr;
  for (const auto& rank_info : exe_desc_.cluster_info()) {
    // init the dns map
    int64_t rank = rank_info.rank();
    std::string ip_port = rank_info.ip_port();
    ss << rank << "\t->\t" << ip_port << "\n";
    rank_to_addr.insert(std::make_pair(rank, ip_port));
    if (rank == cur_rank) {
      addr = ip_port;
    }
  }
  if (addr == "") {
    PADDLE_ENFORCE_EQ(
        rank_to_addr.size(), 1,
        platform::errors::NotFound("Empty address is not valid for "
                                   "paddle.distributed.launch method."));
    PADDLE_ENFORCE_EQ(
        cur_rank, 0,
        platform::errors::NotFound("Address is empty but cur rank is not 0."));
  }
  VLOG(3) << "Current rank is " << cur_rank << " and the ip_port is "
          << (addr == "" ? "empty" : addr) << ".";
  VLOG(3) << "The number of ranks are "
          << (rank_to_addr.size() == 0 ? 1 : rank_to_addr.size()) << ".";
  VLOG(5) << ss.str();
  MessageBus& message_bus_instance = MessageBus::Instance();
  if (!message_bus_instance.IsInit()) {
    message_bus_instance.Init(runtime_graph_->intercepter_id_to_rank(),
                              rank_to_addr, addr);
  }
}

void FleetExecutor::Run() {
  // Run
  Carrier& carrier_instance = Carrier::Instance();
  MessageBus& message_bus_instance = MessageBus::Instance();
  PADDLE_ENFORCE_EQ(
      carrier_instance.IsInit(), true,
      platform::errors::Unavailable("Carrier has not been init yet."));
  PADDLE_ENFORCE_EQ(
      message_bus_instance.IsInit(), true,
      platform::errors::Unavailable("MessageBus has not been init yet."));
  carrier_instance.Start();
}

void FleetExecutor::Release() {
  // Release
}

}  // namespace distributed
}  // namespace paddle
