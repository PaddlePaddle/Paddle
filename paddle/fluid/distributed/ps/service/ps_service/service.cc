/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/distributed/ps/service/ps_service/service.h"

#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <iostream>
#include "paddle/fluid/distributed/ps/service/communicator/communicator.h"
#include "paddle/fluid/string/string_helper.h"

using namespace std;  // NOLINT

namespace paddle {
namespace distributed {

paddle::distributed::PSParameter load_from_prototxt(
    const std::string& filename) {
  paddle::distributed::PSParameter param;
  int file_descriptor = open(filename.c_str(), O_RDONLY);

  if (file_descriptor == -1) {
    VLOG(3) << "FATAL: fail to parse " << filename;
    exit(-1);
  }

  google::protobuf::io::FileInputStream fileInput(file_descriptor);
  if (!google::protobuf::TextFormat::Parse(&fileInput, &param)) {
    VLOG(3) << "FATAL: fail to parse " << filename;
    exit(-1);
  }

  close(file_descriptor);
  return param;
}

void PSCore::InitGFlag(const std::string& gflags) {
  VLOG(3) << "Init With Gflags:" << gflags;
  std::vector<std::string> flags = paddle::string::split_string(gflags);
  if (flags.size() < 1) {
    flags.push_back("-max_body_size=314217728");
    flags.push_back("-socket_max_unwritten_bytes=2048000000");
    flags.push_back("-max_connection_pool_size=1950");
  }
  auto it = flags.begin();
  flags.insert(it, "exe default");
  char* flags_ptr[flags.size()];
  for (size_t i = 0; i < flags.size(); ++i) {
    flags_ptr[i] = (char*)(flags[i].c_str());  // NOLINT
  }
  int params_cnt = flags.size();
  char** params_ptr = &(flags_ptr[0]);
  ::GFLAGS_NAMESPACE::ParseCommandLineFlags(&params_cnt, &params_ptr, true);
}

int PSCore::InitServer(
    const std::string& dist_desc,
    const std::vector<std::string>* host_sign_list, int node_num, int index,
    int trainers,
    const std::vector<framework::ProgramDesc>& server_sub_program) {
  google::protobuf::TextFormat::ParseFromString(dist_desc, &_ps_param);
  InitGFlag(_ps_param.init_gflags());
  _ps_env = paddle::distributed::PaddlePSEnvironment();
  _ps_env.SetPsServers(host_sign_list, node_num);
  _ps_env.SetTrainers(trainers);
  int ret = 0;
  _server_ptr = std::shared_ptr<paddle::distributed::PSServer>(
      paddle::distributed::PSServerFactory::Create(_ps_param));
  ret = _server_ptr->Configure(_ps_param, _ps_env, index, server_sub_program);
  CHECK(ret == 0) << "failed to configure server";
  return ret;
}

int PSCore::InitWorker(
    const std::string& dist_desc,
    const std::map<uint64_t, std::vector<paddle::distributed::Region>>& regions,
    const std::vector<std::string>* host_sign_list, int node_num, int index) {
  google::protobuf::TextFormat::ParseFromString(dist_desc, &_ps_param);
  InitGFlag(_ps_param.init_gflags());
  _ps_env = paddle::distributed::PaddlePSEnvironment();
  _ps_env.SetPsServers(host_sign_list, node_num);
  int ret = 0;
  VLOG(1) << "PSCore::InitWorker";
  auto* communicator = Communicator::GetInstance();
  ret = communicator->GetPsClient()->Configure(_ps_param, regions, _ps_env,
                                               index);
  communicator->Start();
  return ret;
}

std::vector<uint64_t> PSCore::GetClientInfo() {
  return _ps_env.GetClientInfo();
}

int PSCore::CreateClient2ClientConnection(int pserver_timeout_ms,
                                          int pserver_connect_timeout_ms,
                                          int max_retry) {
  int ret = _worker_ptr->CreateClient2ClientConnection(
      pserver_timeout_ms, pserver_connect_timeout_ms, max_retry);
  return ret;
}

uint64_t PSCore::RunServer(const std::string& ip, uint32_t port) {
  return _server_ptr->Start(ip, port);
}

int PSCore::FinalizeWorker() {
  _worker_ptr->FinalizeWorker();
  return 0;
}

int PSCore::StopServer() {
  auto stop_status = _worker_ptr->StopServer();
  stop_status.wait();
  return 0;
}
paddle::distributed::PSParameter* PSCore::GetParam() { return &_ps_param; }
}  // namespace distributed
}  // namespace paddle
