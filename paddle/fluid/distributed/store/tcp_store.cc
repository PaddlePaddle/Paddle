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

#include <poll.h>
#include <chrono>
#include <iostream>
#include <thread>

#include "paddle/fluid/distributed/store/tcp_store.h"
#include "paddle/fluid/distributed/store/tcp_utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

namespace detail {

constexpr int INFTIME = -1;

std::unique_ptr<MasterDaemon> MasterDaemon::start(const Socket& socket) {
  auto master = std::make_unique<MasterDaemon>(socket);
  return master;
}

MasterDaemon::MasterDaemon(const Socket& socket) : _listen_socket(socket) {
  _background_thread = std::thread{&MasterDaemon::run, this};
}

void MasterDaemon::_do_add(int socket) {
  int64_t increment;
  std::string key = tcputils::receive_string(socket);
  tcputils::receive_bytes<int64_t>(socket, &increment, 1);
  int64_t new_value = increment;
  std::vector<uint8_t> old_value;
  auto it = _store.find(key);
  if (it != _store.end()) {
    old_value = it->second;
    char* buffer = reinterpret_cast<char*>(it->second.data());
    size_t len = old_value.size();
    new_value += std::stoll(std::string(buffer, len));
  }

  std::string new_value_str = std::to_string(new_value);
  _store[key] =
      std::vector<uint8_t>(new_value_str.begin(), new_value_str.end());
  tcputils::send_bytes<int64_t>(socket, &new_value, 1);
}

void MasterDaemon::run() {
  std::vector<struct pollfd> fds;
  auto sockfd = _listen_socket.sockfd();
  fds.push_back({.fd = sockfd, .events = POLLIN, .revents = 0});

  bool done = false;
  while (!done) {
    for (size_t i = 0; i < fds.size(); i++) {
      fds[i].revents = 0;
    }

    ::poll(fds.data(), fds.size(), INFTIME);

    if (fds[0].revents != 0) {
      Socket socket = _listen_socket.accept();
      _sockets.emplace_back(socket);
      auto new_sockfd = socket.sockfd();
      fds.push_back({.fd = new_sockfd, .events = POLLIN, .revents = 0});
    }

    for (size_t i = 1; i < fds.size(); i++) {
      if (fds[i].revents == 0) {
        continue;
      }

      Command command;
      tcputils::receive_bytes<Command>(fds[i].fd, &command, 1);

      switch (command) {
        case Command::ADD:
          _do_add(fds[i].fd);
      }
    }
  }
}

std::unique_ptr<TCPServer> TCPServer::create(uint16_t port) {
  auto socket = Socket::listen(port);
  auto server = std::make_unique<TCPServer>();
  server->_master_daemon = MasterDaemon::start(socket);
  return server;
}

std::unique_ptr<TCPClient> TCPClient::connect(
    const std::string host, uint16_t port, const std::chrono::seconds timeout) {
  SocketOptions sock_opt{};
  sock_opt.connect_timeout(timeout);
  const Socket socket = Socket::connect(host, port, sock_opt);
  return std::make_unique<TCPClient>(socket);
}

void TCPClient::send_command_for_key(Command type, const std::string& key) {
  int sockfd = _socket.sockfd();
  tcputils::send_bytes<Command>(sockfd, &type, 1);
  if (key.empty()) {
    return;
  }
  tcputils::send_string(sockfd, key);
}

template <typename T>
void TCPClient::send_value(const T& value) {
  int socket = _socket.sockfd();
  tcputils::send_bytes<T>(socket, &value, 1);
}

template <typename T>
T TCPClient::receive_value() {
  int sockfd = _socket.sockfd();
  T res;
  tcputils::receive_bytes<T>(sockfd, &res, 1);
  return res;
}

}  // namespace detail

TCPStore::TCPStore(std::string host, uint16_t port, bool is_master,
                   size_t num_workers, std::chrono::seconds timeout)
    : Store(timeout), _is_master(is_master), _num_workers(num_workers) {
  if (_is_master) {
    _server = detail::TCPServer::create(port);
  }

  _client = detail::TCPClient::connect(host, port, _timeout);
  waitWorkers();
}

void TCPStore::waitWorkers() {
  if (_num_workers == 0) {
    return;
  }
  add(_init_key, 1);
}

int64_t TCPStore::add(const std::string& key, int64_t value) {
  const std::lock_guard<std::mutex> lock(_lock);
  _client->send_command_for_key(Command::ADD, _key_prefix + key);
  _client->send_value<std::int64_t>(value);
  return _client->receive_value<std::int64_t>();
}

}  // namespace distributed
}  // namespace paddle
