/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#if defined PADDLE_WITH_PSCORE
#include <cstdlib>

#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>  // NOLINT

#include "gtest/gtest.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/distributed/ps/service/heter_client.h"
#include "paddle/fluid/distributed/ps/service/heter_server.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace distributed = paddle::distributed;

PD_DEFINE_string(switch_addr_inner, "127.0.0.1:6000", "addr of inner cluster");
PD_DEFINE_string(switch_addr_heter, "127.0.0.1:6100", "add of inter cluster");
PD_DEFINE_string(peer_switch_addr, "127.0.0.1:7100", "add of inter cluster");

void StartSwitchServer(
    std::shared_ptr<distributed::HeterServer>& switch_server_ptr,  // NOLINT
    std::vector<std::string> endpoints,
    std::vector<std::string> peer_endpoints) {
  switch_server_ptr->SetPeerEndPoints(peer_endpoints);
  switch_server_ptr->SetEndPoint(endpoints[0]);
  switch_server_ptr->StartHeterService(false);
}

void StartSwitchInterServer(
    std::shared_ptr<distributed::HeterServer>& switch_server_ptr,  // NOLINT
    std::vector<std::string> endpoints,
    std::vector<std::string> peer_endpoints) {
  LOG(INFO) << "switch heter service started";
  switch_server_ptr->SetPeerEndPoints(peer_endpoints);
  switch_server_ptr->SetInterEndpoint(endpoints[0]);
  switch_server_ptr->StartHeterInterService(false);
}

int main(int argc, char* argv[]) {
  phi::CPUPlace place;
  phi::CPUContext ctx(place);
  framework::Executor exe(place);

  framework::ProgramDesc program;
  exe.Prepare(program, 0);  // solve undefined symbol: tensor_table.cc

  paddle::flags::ParseCommandLineFlags(&argc, &argv);

  std::string switch_a_endpoint(FLAGS_switch_addr_inner);
  std::string switch_a_endpoint_inter(FLAGS_switch_addr_heter);
  std::string switch_b_endpoint_inter(FLAGS_peer_switch_addr);

  std::shared_ptr<distributed::HeterServer> switch_server_ptr_a =
      std::make_shared<distributed::HeterServer>();

  std::vector<std::string> end_points{switch_a_endpoint};
  std::vector<std::string> peer_endpoints{switch_b_endpoint_inter};
  std::thread switch_server_a_thread(StartSwitchServer,
                                     std::ref(switch_server_ptr_a),
                                     end_points,
                                     peer_endpoints);
  switch_server_ptr_a->WaitServerReady();

  end_points = {switch_a_endpoint_inter};
  peer_endpoints = {switch_b_endpoint_inter};
  std::thread switch_server_a_thread_inter(StartSwitchInterServer,
                                           std::ref(switch_server_ptr_a),
                                           end_points,
                                           peer_endpoints);
  switch_server_ptr_a->WaitServerReady();

  switch_server_a_thread.join();
  LOG(INFO) << "switch_server_a_thread joined";

  switch_server_a_thread_inter.join();
  LOG(INFO) << "switch_server_a_thread_inter joined";

  return 0;
}
#endif
