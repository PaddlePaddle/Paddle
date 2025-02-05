// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef _WIN32
#include <gloo/common/win.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#include <array>
#endif

#include <cstdlib>
#include <cstring>

#include "paddle/common/errors.h"
#include "paddle/phi/core/distributed/gloo_utils.h"
#include "paddle/phi/core/distributed/store/tcp_utils.h"
#include "paddle/phi/core/enforce.h"

namespace phi::distributed {
std::shared_ptr<gloo::transport::Device> CreateDeviceForInterface(
    const std::string& ifname) {
  gloo::transport::tcp::attr attr;
  attr.iface = ifname;
  return gloo::transport::tcp::CreateDevice(attr);
}

std::shared_ptr<gloo::transport::Device> CreateDeviceForHostname(
    const std::string& hostname) {
  gloo::transport::tcp::attr attr;
  attr.hostname = hostname;
  return gloo::transport::tcp::CreateDevice(attr);
}

std::shared_ptr<gloo::transport::Device> CreateDefaultDevice() {
  std::array<char, HOST_NAME_MAX> hostname = {};
  auto ret = ::gethostname(hostname.data(), HOST_NAME_MAX);
  PADDLE_ENFORCE_EQ(
      ret,
      0,
      common::errors::Fatal("Get hostname error for createDefaultDevice."));
  ::addrinfo* result;
  result = phi::distributed::tcputils::get_addr_info(
      hostname.data(), "", 0, AF_UNSPEC);
  ::addrinfo* cur;
  for (cur = result; cur != nullptr; cur = cur->ai_next) {
    phi::distributed::SocketType socket =
        ::socket(cur->ai_family, cur->ai_socktype, cur->ai_protocol);
    if (socket == -1) {
      continue;
    }
    ret = ::bind(socket, cur->ai_addr, cur->ai_addrlen);
#ifdef _WIN32
    closesocket(socket);
#else
    close(socket);
#endif
    if (ret == -1) {
      continue;
    }
    break;
  }
  freeaddrinfo(result);
  if (cur != nullptr) {
    return CreateDeviceForHostname(hostname.data());
  }
  return CreateDeviceForHostname("127.0.0.1");
}

std::shared_ptr<gloo::transport::Device> CreateGlooDevice() {
  char* ifname = std::getenv("GLOO_SOCKET_IFNAME");
  if (ifname && std::strlen(ifname) > 1) {
    return CreateDeviceForInterface(std::string(ifname));
  } else {
    return CreateDefaultDevice();
  }
}

void send_recv(SendRecvOptions* opts) {
  const auto& context = opts->context;
  gloo::transport::UnboundBuffer* in = opts->in.get();
  gloo::transport::UnboundBuffer* out = opts->out.get();
  const auto slot = gloo::Slot::build(kSendRecvSlotPrefix, opts->tag);

  if (context->rank == opts->src) {
    in->send(opts->dst, slot);
    in->waitSend(opts->timeout);
  } else if (context->rank == opts->dst) {
    out->recv(opts->src, slot);
    out->waitRecv(opts->timeout);
  }
}

}  // namespace phi::distributed
