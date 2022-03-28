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
#include <cerrno>
#include <cstring>
#include <thread>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {
namespace tcputils {

std::error_code socket_error() {
#ifdef _WIN32
  return std::error_code{::WSAGetLastError(), std::generic_category()};
#else
  return std::error_code{errno, std::generic_category()};
#endif
}

void close_socket(SocketType socket) {
#ifdef _WIN32
  ::closesocket(socket);
#else
  ::close(socket);
#endif
}

::addrinfo* get_addr_info(const std::string host, const std::string port,
                          int ai_flags, int family) {
  ::addrinfo hints{}, *res;
  hints.ai_flags = ai_flags;
  hints.ai_family = family;
  hints.ai_socktype = SOCK_STREAM;

  const char* node = host.empty() ? nullptr : host.c_str();
  const char* port_cstr = port.empty() ? nullptr : port.c_str();

  int n;
  n = ::getaddrinfo(node, port_cstr, &hints, &res);
  const char* gai_err = ::gai_strerror(n);
  const char* proto =
      (family == AF_INET ? "IPv4" : family == AF_INET6 ? "IPv6" : "");
  PADDLE_ENFORCE_EQ(
      n, 0, platform::errors::InvalidArgument(
                "%s network %s:%s cannot be obtained. Details: %s.", proto,
                host, port, gai_err));

  return res;
}

void free_addr_info(::addrinfo* hint) {
  PADDLE_ENFORCE_NOT_NULL(
      hint, platform::errors::InvalidArgument(
                "The parameter for free_addr_info cannot be null."));
  ::freeaddrinfo(hint);
}

SocketType tcp_connect(const std::string host, const std::string port,
                       int family, std::chrono::seconds timeout) {
  int ai_flags = AI_NUMERICSERV | AI_V4MAPPED | AI_ALL;
  ::addrinfo* res = get_addr_info(host, port, ai_flags, family);

  SocketType sockfd = -1;
  bool retry = true;
  auto deadline = std::chrono::steady_clock::now() + timeout;
  do {
    for (::addrinfo* cur = res; cur != nullptr; cur = cur->ai_next) {
      sockfd = ::socket(cur->ai_family, cur->ai_socktype, cur->ai_protocol);
      PADDLE_ENFORCE_GT(sockfd, 0, platform::errors::InvalidArgument(
                                       "Create socket to connect %s:%s failed. "
                                       "Details: %s. ",
                                       host, port, socket_error().message()));

      if (::connect(sockfd, cur->ai_addr, cur->ai_addrlen) == 0) {
        retry = false;
        break;
      }
      VLOG(0) << "Retry to connect to " << host << ":" << port
              << " while the server is not yet listening.";
      close_socket(sockfd);
      sockfd = -1;
      std::this_thread::sleep_for(kDelay);
      if (timeout != kNoTimeout &&
          std::chrono::steady_clock::now() >= deadline) {
        retry = false;
        break;
      }
    }

    if (timeout != kNoTimeout && std::chrono::steady_clock::now() >= deadline) {
      retry = false;
    }
  } while (retry);

  free_addr_info(res);

  PADDLE_ENFORCE_GT(sockfd, 0,
                    platform::errors::InvalidArgument(
                        "Network %s:%s cannot be connected.", host, port));
  VLOG(0) << "Successfully connected to " << host << ":" << port;

  return sockfd;
}

SocketType tcp_listen(const std::string host, const std::string port,
                      int family) {
  int ai_flags = AI_PASSIVE | AI_NUMERICSERV;
  ::addrinfo* res = get_addr_info(host, port, ai_flags, family);
  ::addrinfo* cur = res;
  SocketType sockfd{};

  std::string node = host.empty() ? "IP_ANY" : host;
  while (cur) {
    sockfd = ::socket(cur->ai_family, cur->ai_socktype, cur->ai_protocol);
    if (sockfd < 0) {
      VLOG(0) << "Cannot create socket on " << node << ":" << port
              << ". Details: " << socket_error().message();
      cur = cur->ai_next;
      continue;
    }

    int on = 1;
#ifdef _WIN32
    int ret = ::setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR,
                           reinterpret_cast<char*>(&on), sizeof(on));
#else
    int ret = ::setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
#endif
    if (ret < 0) {
      VLOG(0) << "Set the address reuse option failed on the server.";
    }
    if (::bind(sockfd, res->ai_addr, res->ai_addrlen) == 0) {
      break;
    }
    close_socket(sockfd);
    sockfd = -1;
    cur = cur->ai_next;
  }

  PADDLE_ENFORCE_GT(sockfd, 0,
                    platform::errors::InvalidArgument(
                        "Bind network on %s:%s failedd.", node, port));

  ::listen(sockfd, LISTENQ);

  VLOG(0) << "The server starts to listen on " << node << ":" << port;
  return sockfd;
}

SocketType tcp_accept(SocketType socket) {
  ::sockaddr_storage addr_s{};
  ::socklen_t addr_len = sizeof(addr_s);
  SocketType new_socket =
      ::accept(socket, reinterpret_cast<::sockaddr*>(&addr_s), &addr_len);
  PADDLE_ENFORCE_GT(
      new_socket, 0,
      platform::errors::InvalidArgument(
          "The server failed to accept a new connection. Details: %s.",
          socket_error().message()));
#ifndef _WIN32
  ::fcntl(new_socket, F_SETFD, FD_CLOEXEC);
#endif
  auto value = 1;
#ifdef _WIN32
  ::setsockopt(new_socket, IPPROTO_TCP, TCP_NODELAY,
               reinterpret_cast<const char*>(&value), sizeof(value));
#else
  ::setsockopt(new_socket, IPPROTO_TCP, TCP_NODELAY, &value, sizeof(value));
#endif
  return new_socket;
}

void send_string(SocketType socket, const std::string& s) {
  std::string::size_type size = s.size();
  send_bytes<std::string::size_type>(socket, &size, 1);
  send_bytes<const char>(socket, s.data(), size);
}

std::string receive_string(SocketType socket) {
  std::string::size_type size;
  receive_bytes<std::string::size_type>(socket, &size, 1);
  std::vector<char> v(size);
  receive_bytes<char>(socket, v.data(), size);
  return std::string(v.data(), v.size());
}

}  // namespace tcputils
}  // namespace distributed
}  // namespace paddle
