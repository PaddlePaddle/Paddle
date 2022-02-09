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

#include "paddle/fluid/distributed/store/tcp_utils.h"
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {
namespace tcputils {

using TimePoint = std::chrono::time_point<std::chrono::system_clock>;

std::error_code getSocketError() {
  return std::error_code{errno, std::generic_category()};
}

::addrinfo* getAddrInfo(const std::string host, const std::string service,
                        int ai_flags, int family) {
  ::addrinfo hints{}, *res;
  hints.ai_flags = ai_flags;
  hints.ai_family = family;
  hints.ai_socktype = SOCK_STREAM;

  int n;
  n = ::getaddrinfo(host.c_str(), service.c_str(), &hints, &res);
  const char* gai_err = ::gai_strerror(n);
  const char* network =
      (family == AF_INET ? "IPv4" : family == AF_INET6 ? "IPv6" : "");
  PADDLE_ENFORCE_EQ(
      n, 0, platform::errors::InvalidArgument(
                "Local network %s cannot be obtained. Details: %s.", network,
                gai_err));

  return res;
}

void freeAddrInfo(::addrinfo* info) {
  PADDLE_ENFORCE_NOT_NULL(
      info, platform::errors::InvalidArgument(
                "The pointer to free for freeAddrInfo should not be null."));
  ::freeaddrinfo(info);
}

int tcpConnect(const std::string host, const std::string service, int family,
               std::chrono::milliseconds timeout) {
  int ai_flags = AI_NUMERICSERV | AI_V4MAPPED | AI_ALL;
  ::addrinfo* res = getAddrInfo(host, service, ai_flags, family);

  int sockfd;
  bool is_timeout = false;
  auto deadline = std::chrono::steady_clock::now() + timeout;
  do {
    sockfd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sockfd < 0) {
      VLOG(0) << "Trying to connect to " << host << ":" << service
              << " failed. Details: " << std::strerror(errno);
      continue;
    }

    if (::connect(sockfd, res->ai_addr, res->ai_addrlen) == 0) {
      break;
    }

    if (timeout != kNoTimeOut && std::chrono::steady_clock::now() >= deadline) {
      is_timeout = true;
    }
  } while (res->ai_next && !is_timeout);

  PADDLE_ENFORCE_GT(sockfd, 0, platform::errors::InvalidArgument(
                                   "Local network %s:%s cannot be connected.",
                                   host, service));

  return sockfd;
}

int tcpListen(const std::string host, const std::string service, int family) {
  int ai_flags = AI_PASSIVE | AI_NUMERICSERV;
  ::addrinfo* res = getAddrInfo(host, service, ai_flags, family);
  int sockfd;

  do {
    sockfd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sockfd < 0) {
      VLOG(0) << "Cannot create socket on " << host << ":" << service
              << " Details: " << std::strerror(errno);
      continue;
    }

    const int on = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) < 0) {
      VLOG(0) << "Cannot set the address reuse option on " << host << ":"
              << service;
    }
    if (::bind(sockfd, res->ai_addr, res->ai_addrlen) == 0) {
      break;
    }
  } while (res->ai_next);

  PADDLE_ENFORCE_GT(
      sockfd, 0, platform::errors::InvalidArgument(
                     "Local network %s:%s cannot be binded.", host, service));

  ::listen(sockfd, LISTENQ);

  VLOG(0) << "The server starts to listen on " << host << ":" << service;

  return sockfd;
}

int tcpAccept(int sock) {
  ::sockaddr_storage addr_s{};
  ::socklen_t addr_len = sizeof(addr_s);
  int new_sock =
      ::accept(sock, reinterpret_cast<::sockaddr*>(&addr_s), &addr_len);
  PADDLE_ENFORCE_GT(
      new_sock, 0,
      platform::errors::InvalidArgument(
          "The server failed to accept a new connection. Details: %s.",
          std::strerror(errno)));
  ::fcntl(new_sock, F_SETFD, FD_CLOEXEC);
  auto value = 1;
  ::setsockopt(new_sock, IPPROTO_TCP, TCP_NODELAY, &value, sizeof(value));
  return new_sock;
}

void setSockOpt(int sock, int level, int optname, const char* value,
                int optlen) {
  ::setsockopt(sock, level, optname, value, optlen);
}

void sendString(int sock, const std::string& s) {
  size_t size = s.size();
  sendBytes<size_t>(sock, &size, 1);
  sendBytes<char>(sock, s.data(), size);
}

std::string recvString(int sock) {
  size_t size;
  recvBytes<size_t>(sock, &size, 1);
  std::vector<char> v(size);
  sendBytes<char>(sock, v.data(), size);
  return std::string(v.data(), v.size());
}

}  // namespace tcputils
}  // namespace distributed
}  // namespace paddle
