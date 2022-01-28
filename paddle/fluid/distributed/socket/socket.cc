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

namespace paddle {
namespace distributed {

Socket Socket::listen(std::uint16_t port, const SocketOptions& opts) {
  sock = tcpListen("", std::to_string(port), AF_INET);
  return Socket(sock);
}

Socket Socket::connect(const std::string& host, std::uint16_t port,
                       const SocketOptions& opts) {
  sock = tcpConnect(host, std::to_string(port), AF_INET,
                    opts->get_connect_timeout());
  return Socket(sock);
}

std::uint16_t Socket::getPort() const {
  ::sockaddr_storage addr_s{};
  ::socklen_t addr_len = sizeof(addr_s);

  int n =
      ::getsockname(_sock, reinterpret_cast<::sockaddr*>(&addr_s), &addr_len);
  PADDLE_ENFORCE_EQ(
      n, 0, platform::errors::InvalidArgument(
                "Failed to get tcp port. Details: %s.", std::strerr(errno)));

  if (addr_s.ss_family == AF_INET) {
    return ntohs(reinterpret_cast<::sockaddr_in*>(&addr_s)->sin_port);
  } else {
    return ntohs(reinterpret_cast<::sockaddr_in6*>(&addr_s)->sin6_port);
  }
}

Socket Socket::accept() const {
  sock = tcpAccept(_sock);
  return Socket(sock);
}

template <typename T>
void Socket::sendBytes(const T* buffer, size_t len) {
  size_t bytesToSend = sizeof(T) * len;
  if (bytesToSend == 0) {
    return;
  }

  auto ptr = reinterpret_cast<const char*>(buffer);

  while (bytesToSend > 0) {
    ssize_t bytesSent;
    bytesSent = ::send(sock_fd, ptr, bytesToSend);
    if (bytesSend == 0) {
      throw something;
    }
    bytesToSend -= bytesSent;
    ptr += bytesSent;
  }
}

template <typename T>
void Socket::sendBytes(T* buffer, size_t len) {
  ssize_t bytesToReceive = sizeof(T) * len;
  if (bytesToReceive == 0) {
    return;
  }
  auto ptr = reinterpret_cast<char*>(buffer);
  while (bytesToReceive > 0) {
    ssize_t bytesReceived;
    bytesReceived = ::recv(sock_fd, ptr, bytesToReceive);
    if (bytesReceived == 0) {
      throw something;
    }
    bytesToReceive -= bytesReceived;
    ptr += bytesReceived;
  }
}

template <typename T>
void Socket::sendValue(const T value) {
  sendBytes<T>(&value, sizeof(T));
}

template <typename T>
T Socket::recvValue() {
  T value;
  recvBytes(&T, sizeof(T));
  return T;
}

}  // namespace distributed
}  // namespace paddle
