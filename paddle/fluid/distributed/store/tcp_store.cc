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

std::unique_ptr<MasterDaemon> MasterDaemon::start(Socket sock) {
  auto master = std::make_unique<MasterDaemon>(sock);
  return master;
}

MasterDaemon::MasterDaemon(Socket socket) : _socket(socket) {
  _background_thread = std::thread{&MasterDaemon::run, this};
}

void MasterDaemon::doGet(int sock) {
  // std::string key = tcputils::recvString(sock);
  // auto data = _store.at(key);
  // tcputils::sendVector<uint8_t>(socket, data);
}

void MasterDaemon::doSet(int sock) {}

void MasterDaemon::doRemove(int sock) {}

void MasterDaemon::doAdd(int sock) {
  VLOG(0) << "doAdd:here1";
  std::string key = tcputils::recvString(sock);
  int64_t incr = tcputils::recvValue<int64_t>(sock);
  int64_t new_value = incr;
  std::vector<uint8_t> old_value;
  auto it = _store.find(key);
  if (it != _store.end()) {
    old_value = it->second;
    char* buffer = reinterpret_cast<char*>(it->second.data());
    size_t len = old_value.size();
    new_value += std::stoll(std::string(buffer, len));
  }
  VLOG(0) << "doAdd:here2";

  std::string new_value_str = std::to_string(new_value);
  _store[key] =
      std::vector<uint8_t>(new_value_str.begin(), new_value_str.end());
  tcputils::sendValue<int64_t>(sock, new_value);
  VLOG(0) << "doAdd:here3";
}

void MasterDaemon::doWait(int sock) {}

void MasterDaemon::run() {
  std::vector<struct pollfd> fds;
  auto sock_fd = _socket.get_socket_fd();
  struct pollfd listen_fd;
  listen_fd.fd = sock_fd;
  listen_fd.events = POLLIN | POLLRDNORM;
  // fds.push_back({.fd = sock_fd, .events = POLLIN, .revents = 0});
  fds.push_back(listen_fd);

  bool done = false;
  while (!done) {
    // for(size_t i = 0; i < fds.size(); i++) {
    //  fds[i].revents = 0;
    //}
    VLOG(0) << "run:here1_0_3";
    VLOG(0) << "run:fds[0]: " << fds[0].fd;
    VLOG(0) << "run:fds[0].events: " << fds[0].events;
    ::poll(fds.data(), fds.size(), -1);
    VLOG(0) << "run:here1_0_4";

    if (fds[0].revents != 0) {
      Socket socket = _socket.accept();
      VLOG(0) << "run:here1_0_5";
      _sockets.emplace_back(socket);
      auto new_sock_fd = socket.get_socket_fd();
      fds.push_back({.fd = new_sock_fd, .events = POLLIN, .revents = 0});
    }

    for (size_t i = 1; i < fds.size(); i++) {
      if (fds[i].revents == 0) {
        VLOG(0) << "run:here1_1";
        continue;
      }

      Command command;
      VLOG(0) << "run:here1_2";
      tcputils::recvBytes<Command>(fds[i].fd, &command, 1);
      VLOG(0) << "run:here2";

      switch (command) {
        case Command::GET:
          doGet(fds[i].fd);
        case Command::SET:
          doSet(fds[i].fd);
        case Command::REMOVE:
          doRemove(fds[i].fd);
        case Command::ADD:
          VLOG(0) << "run:here3";
          doAdd(fds[i].fd);
        case Command::WAIT:
          doWait(fds[i].fd);
          // default:
          //  PADDLE_THROW("Unknown command received %d.", command);
      }
    }
  }
}

std::unique_ptr<TCPServer> TCPServer::create(uint16_t port) {
  auto socket = Socket::listen(port);
  auto server = std::make_unique<TCPServer>(port);
  server->_master_daemon = MasterDaemon::start(socket);
  return server;
}

std::unique_ptr<TCPClient> TCPClient::connect(
    const std::string host, uint16_t port,
    const std::chrono::milliseconds& timeout) {
  SocketOptions sock_opt{};
  sock_opt.set_connect_timeout(timeout);
  const Socket socket = Socket::connect(host, port, sock_opt);
  return std::make_unique<TCPClient>(socket);
}

void TCPClient::sendCommandForKey(Command type, const std::string& key) {
  int sock_fd = _socket.get_socket_fd();
  VLOG(0) << "client socket: " << sock_fd;
  tcputils::sendValue<Command>(sock_fd, type);
  if (key.empty()) {
    return;
  }
  tcputils::sendString(sock_fd, key);
}

void TCPClient::sendCommand(Command type) { sendCommandForKey(type, ""); }

void TCPClient::sendBytes(const std::vector<std::uint8_t>& bytes) {
  int sock_fd = _socket.get_socket_fd();
  tcputils::sendVector<std::uint8_t>(sock_fd, bytes);
}

std::vector<uint8_t> TCPClient::recvBytes() {
  int sock_fd = _socket.get_socket_fd();
  return tcputils::recvVector<std::uint8_t>(sock_fd);
}

template <typename T>
void TCPClient::sendValue(const T& value) {
  int sock_fd = _socket.get_socket_fd();
  tcputils::sendValue<T>(sock_fd, value);
}

template <typename T>
T TCPClient::recvValue() {
  int sock_fd = _socket.get_socket_fd();
  return tcputils::recvValue<T>(sock_fd);
}

void TCPClient::sendStrings(std::vector<std::string> strings) {
  size_t size = strings.size();
  int sock_fd = _socket.get_socket_fd();
  tcputils::sendValue<size_t>(sock_fd, size);

  if (strings.empty()) {
    return;
  }

  for (auto& s : strings) {
    tcputils::sendString(sock_fd, s);
  }
}

}  // namespace detail

TCPStore::TCPStore(std::string host, uint16_t port, bool is_master,
                   size_t num_workers, std::chrono::milliseconds timeout)
    : Store(timeout),
      _host(host),
      _port(port),
      _is_master(is_master),
      _num_workers(num_workers) {
  if (_is_master) {
    _server = detail::TCPServer::create(port);
  }

  _client = detail::TCPClient::connect(_host, _port, _timeout);
  waitWorkers();
}

void TCPStore::waitWorkers() {
  if (_num_workers == 0) {
    return;
  }

  VLOG(0) << "here0";
  add(_init_key, 1);
  VLOG(0) << "here0_1";
  auto begin = std::chrono::steady_clock::now();
  if (_server) {
    do {
      VLOG(0) << "here1";
      auto value = get(_init_key);
      int completed = std::stoi(std::string(value.begin(), value.end()));
      VLOG(0) << "completed: " << completed;
      if (completed >= _num_workers) {
        VLOG(0) << "here2";
        break;
      }
      const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - begin);
      if (_timeout != kNoTimeout && elapsed > _timeout) {
        break;
      }
    } while (true);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void TCPStore::set(const std::string& key, const std::vector<uint8_t>& value) {
  const std::lock_guard<std::mutex> lock(_lock);
  _client->sendCommandForKey(Command::SET, _key_prefix + key);
  _client->sendBytes(value);
}

std::vector<uint8_t> TCPStore::get(const std::string& key) {
  const std::lock_guard<std::mutex> lock(_lock);
  doWait(std::vector<std::string>{key}, _timeout);
  _client->sendCommandForKey(Command::GET, _key_prefix + key);
  return _client->recvBytes();
}

int64_t TCPStore::add(const std::string& key, int64_t value) {
  VLOG(0) << "add::here1";
  const std::lock_guard<std::mutex> lock(_lock);
  VLOG(0) << "add::here2";
  _client->sendCommandForKey(Command::ADD, _key_prefix + key);
  VLOG(0) << "add::here3";
  _client->sendValue<std::int64_t>(value);
  VLOG(0) << "add::here4";
  return _client->recvValue<std::int64_t>();
}

bool TCPStore::removeKey(const std::string& key) {
  const std::lock_guard<std::mutex> lock(_lock);
  _client->sendCommandForKey(Command::REMOVE, _key_prefix + key);
  auto num_removed = _client->recvValue<std::int64_t>();
  return num_removed == 1;
}

void TCPStore::wait(const std::vector<std::string>& keys) {
  wait(keys, _timeout);
}

void TCPStore::wait(const std::vector<std::string>& keys,
                    const std::chrono::milliseconds& timeout) {
  std::vector<std::string> keys_with_prefix;
  for (auto key : keys) {
    keys_with_prefix.emplace_back(_key_prefix + key);
  }
  doWait(keys_with_prefix, timeout);
}

void TCPStore::doWait(const std::vector<std::string>& keys,
                      const std::chrono::milliseconds& timeout) {
  // auto pre_timeout = _client->getTimeout();
  _client->sendCommand(Command::WAIT);
  _client->sendStrings(keys);

  // _client->setTimeout(timeout);
  auto reply = _client->recvValue<WaitReplyType>();
  PADDLE_ENFORCE_EQ(static_cast<int>(reply),
                    static_cast<int>(WaitReplyType::STOP_WAIT),
                    platform::errors::InvalidArgument(
                        "The call to doWait must reply with STOP_WAITING."));
  // _client->setTimeout(pre_timeout);
}

}  // namespace distributed
}  // namespace paddle
