/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/distributed/ps.pb.h"
#include "paddle/fluid/distributed/ps/service/ps_client.h"
#include "paddle/fluid/distributed/ps/service/sendrecv.pb.h"
#include "paddle/fluid/distributed/ps/service/server.h"

namespace paddle {
namespace distributed {

class PSClient;
class PSServer;
class PsRequestMessage;
class PsResponseMessage;
class PsService;

using paddle::distributed::PsRequestMessage;
using paddle::distributed::PsResponseMessage;
using paddle::distributed::PsService;

class PSCore {
 public:
  explicit PSCore() {}
  virtual ~PSCore() {}

  virtual int init_server(
      const std::string& dist_desc,
      const std::vector<std::string>* host_sign_list, int node_num, int index,
      int trainers,
      const std::vector<framework::ProgramDesc>& server_sub_program = {});
  virtual int init_worker(
      const std::string& dist_desc,
      const std::map<uint64_t, std::vector<paddle::distributed::Region>>&
          regions,
      const std::vector<std::string>* host_sign_list, int node_num, int index);
  virtual uint64_t run_server(const std::string& ip, uint32_t port);
  virtual int stop_server();
  virtual int finalize_worker();
  virtual std::vector<uint64_t> get_client_info();
  virtual int create_client2client_connection(int pserver_timeout_ms,
                                              int pserver_connect_timeout_ms,
                                              int max_retry);
  std::shared_ptr<paddle::distributed::PSServer>
      _server_ptr;  // pointer to server
  std::shared_ptr<paddle::distributed::PSClient>
      _worker_ptr;  // pointer to worker
  virtual paddle::distributed::PSParameter* get_param();

 private:
  void init_gflag(const std::string& gflags);
  paddle::distributed::PSParameter _ps_param;
  paddle::distributed::PaddlePSEnvironment _ps_env;
};

}  // namespace distributed
}  // namespace paddle
