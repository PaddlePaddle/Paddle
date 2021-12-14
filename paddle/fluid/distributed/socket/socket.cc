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

#include <socket.h>

namespace paddle {
namespace distributed {

bool Socket::tryListen(const ::addrinfo& addr) {
  sock_fd = ::socket(addr.ai_family, addr.ai_socktype, addr.ai_protocol);
  if (sock_fd == -1) {
    return false;
  }

  if (::bind(sock_fd, addr.ai_addr, addr.ai_addrlen) != 0) {
    return false;
  }

  // TODO(sandyhouse) a more smart way to set backlog.
  if (::listen(sock_fd, /*backlog=*/1024) != 0) {
    return false;
  }
  return true;
}

bool Socket::tryListen(std::uint16_t port) {
  ::addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC;  // IPv4 and IPv6
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;

  std::string port_str = std::to_string(port)

      int ret = ::getaddrinfo(nullptr, port_str.c_str(), &hints, &res);
  if (ret != 0) {
    const char* err_str = ::gai_strerror(ret);
    VLOG(1) << "The network address cannot be retrieved: " << ret << " - "
            << err_str;
    return false;
  }

  for (::addrinfo* addr = res; addr != nullptr; addr = addr->ai_next) {
    if (tryListen(*addr)) {
      return true;
    }
  }
  return false;
}

void Socket::listen(std::uint16_t port) {
  if (tryListen(port)) {
    return;
  }
}

bool Socket::tryConnect(const ::addrinfo& addr) {
  sockfd = ::socket(addr.ai_family, addr.ai_socktype, addr.ai_protocol);
  if (sock_fd == -1) {
    return false;
  }
  int ret = ::connect(sock_fd, addr.ai_addr, addr.ai_addrlen);
  if (ret == 0) {
    return true;
  }
  std::error_code err = errno;
  if (err == std::errc::already_connected) {
    return true;
  }
  if (err != std::errc::operation_would_block &&
      err != std::errc::operation_in_progress) {
    // return false and set retry to false;
  }

  ::pollfd pfd{};
  pfd.fd = sock_fd;
  pfd.events = POLLOUT;

  ret = pollFd(&pfd, 1);
  if (ret == 0) {
    return timeout;
  }
  if (ret == -1) {
    return error;
  }
}

void Socket::tryConnect(const std::string& host, std::uint16_t port) {
  ::addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC;  // IPv4 and IPv6
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  hints.ai_flags = AI_ALL | AI_V4MAPPED | AI_NUMERICSERV;

  std::string port_str = std::to_string(port)

      int ret = ::getaddrinfo(host, port_str.c_str(), &hints, &res);
  if (ret != 0) {
    const char* err_str = ::gai_strerror(ret);
    VLOG(1) << "The network address cannot be retrieved: " << ret << " - "
            << err_str;
    return false;
  }

  bool retry = true;
  do {
    for (::addrinfo* addr = res; addr != nullptr; addr = addr->ai_next) {
      if (tryConnect(*addr)) {
        return true;
      }
    }
    if (> timeout) {
      retry = false;
      // sleep
      ::timespec ts{};
      ts.tv_sec = d.count();
      if (::nanosleep(&ts, nullptr) != 0) {
        std::error_code err = errno;
        if (err == std::errc::interrupted) {
          throw some;
        }
      }
    }
  } while (retry);
  return false;
}

void Socket::connect(const std::string& host, std::uint16_t port) {
  if (tryConnect(host, port)) {
    return;
  }
}

std::uint16_t Socket::getPort() const {
  ::sockaddr_storage addr_s{};
  ::socklen_t addr_len = sizeof(addr_s);

  if (::getsockname(sock_fd, reinterpret_cast<::sockaddr*>(&addr_s),
                    &addr_len) != 0) {
    throw some;
  }
  if (addr_s.ss_family == AF_INET) {
    return ntohs(reinterpret_cast<::sockaddr_in*>(&addr_s)->sin_port);
  } else {
    return ntohs(reinterpret_cast<::sockaddr_in6*>(&addr_s)->sin6_port);
  }
}

Socket Socket::accept() const {
  ::sockaddr_storage addr_s{};
  ::socklen_t addr_len = sizeof(addr_s);
  int fd = ::accept(sock_fd, reinterpret_cast<::sockaddr*>(&addr_s), &addr_len);
  if (fd == -1) {
    throw some;
  }
  ::addrinfo addr{};
  addr.ai_addr = reinterpret_cast<::sockaddr*>(&addr_s);
  addr.ai_addrlen = addr_len;
  return Socket(fd);
}

}  // namespace distributed
}  // namespace paddle
