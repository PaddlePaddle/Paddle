// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/store/socket.h"
#include <sys/socket.h>
#include <sys/types.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include "paddle/fluid/distributed/store/tcp_utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

Socket Socket::listen(std::uint16_t port) {
  int socket = tcputils::tcp_listen("", std::to_string(port), AF_INET);
  return Socket(socket);
}

Socket Socket::connect(const std::string& host, std::uint16_t port,
                       const SocketOptions& opts) {
  int socket = tcputils::tcp_connect(host, std::to_string(port), AF_INET,
                                     opts.connect_timeout());
  return Socket(socket);
}

Socket Socket::accept() const {
  auto socket = tcputils::tcp_accept(_sockfd);
  return Socket(socket);
}

}  // namespace distributed
}  // namespace paddle
