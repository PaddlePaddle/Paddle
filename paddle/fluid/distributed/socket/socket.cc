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

#include "paddle/fluid/distributed/socket/socket.h"
#include <sys/socket.h>
#include <sys/types.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

Socket Socket::listen(std::uint16_t port, const SocketOptions& opts) {
  int sock = tcputils::tcpListen("", std::to_string(port), AF_INET);
  return Socket(sock);
}

Socket Socket::connect(const std::string& host, std::uint16_t port,
                       const SocketOptions& opts) {
  int sock = tcputils::tcpConnect(host, std::to_string(port), AF_INET,
                                  opts.get_connect_timeout());
  return Socket(sock);
}

std::uint16_t Socket::getPort() const {
  ::sockaddr_storage addr_s{};
  ::socklen_t addr_len = sizeof(addr_s);

  int n =
      ::getsockname(_sock, reinterpret_cast<::sockaddr*>(&addr_s), &addr_len);
  PADDLE_ENFORCE_EQ(
      n, 0, paddle::platform::errors::InvalidArgument(
                "Failed to get tcp port. Details: %s.", std::strerror(errno)));

  if (addr_s.ss_family == AF_INET) {
    return ntohs(reinterpret_cast<::sockaddr_in*>(&addr_s)->sin_port);
  } else {
    return ntohs(reinterpret_cast<::sockaddr_in6*>(&addr_s)->sin6_port);
  }
}

Socket Socket::accept() const {
  auto sock = tcputils::tcpAccept(_sock);
  return Socket(sock);
}

template <typename T>
void Socket::sendBytes(const T* buffer, size_t len) {
  tcputils::sendBytes<T>(_sock, buffer, len);
}

template <typename T>
void Socket::recvBytes(T* buffer, size_t len) {
  tcputils::recvBytes<T>(_sock, buffer, len);
}

template <typename T>
void Socket::sendValue(const T& value) {
  tcputils::sendValue<T>(_sock, value);
}

template <typename T>
T Socket::recvValue() {
  return tcputils::recvValue<T>(_sock);
}

}  // namespace distributed
}  // namespace paddle
