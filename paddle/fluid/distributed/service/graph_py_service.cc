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

#include "paddle/fluid/distributed/service/graph_py_service.h"
namespace paddle {
namespace distributed {
std::vector<std::string> graph_service::split(std::string &str,
                                              const char pattern) {
  std::vector<std::string> res;
  std::stringstream input(str);
  std::string temp;
  while (std::getline(input, temp, pattern)) {
    res.push_back(temp);
  }
  return res;
}
void graph_service::init(std::string ips_str, int shard_num) {
  std::istringstream stream(ips_str);
  std::string ip, port;
  server_size = 0;
  std::vector<std::string> ips_list = split(ips_str, ';');
  int index = 0;
  for (auto ips : ips_list) {
    auto ip_and_port = split(ips, ':');
    server_list.push_back(ip_and_port[0]);
    port_list.push_back(ip_and_port[1]);
    // auto ph_host = paddle::distributed::PSHost(ip_and_port[0],
    // ip_and_port[1], index);
    // host_sign_list_.push_back(ph_host.serialize_to_string());
    index++;
  }
  start_client();
}
}
}