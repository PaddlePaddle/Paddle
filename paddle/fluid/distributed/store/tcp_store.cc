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

#pragma once

#include <iostream>

#include "paddle/fluid/distributed/store/tcp_store.h"

namespace paddle {
namespace distributed {

namespace detail {

class TCPServer {
 public:
  explicit TCPServer(uint16_t port) : _port(port) {}
  static std::shared_ptr<TCPServer> start(std::uint16_t port);

 private:
  std::uint16_t _port;
}

std::shared_ptr<TCPServer>
TCPServerstart(std::uint16_t port) {
  std::shared_ptr<TCPServer> server{};
  Socket socket = Socket::listen(port);
  return std::make_shared<TCPServer>(port);
}

class TCPClient {
 public:
  explicit TCPClient(Socket&& socket) : _socket{std::move(socket)} {}
  static std::unique_ptr<TCPClient> connect(
      const std::string host, uint16_t port,
      const std::chrono::seconds& timeout);
  void sendCommandForKey(CommandType type, const std::string& key);
  void sendStrings(std::vector<std::string> strings);
  void sendBytes(const std::vector<std::uint8_t>& bytes);
  std::vector<uint8_t> recvBytes() {
    int sock_fd = _socket.get_socket_fd();
    return tcputil::recvVector<std::uint8_t>(sock_fd);
  }

  template <typename T>
  void sendValue(const T& value) {
    int sock_fd = _socket.get_socket_fd();
    tcputil::sendValue<T>(sock_fd, value);
  }

  template <typename T>
  T recvValue(const T& value) {
    int sock_fd = _socket.get_socket_fd();
    return tcputil::recvValue<T>(sock_fd);
  }

 private:
  Socket _socket;
}

std::unique_ptr<TCPClient>
TCPClient::connect(const std::string host, uint16_t port,
                   const std::chrono::seconds& timeout) {
  Socket socket =
      Socket::connect(host, port, SocketOptions{}.set_timeout(timeout));
  return std::make_unique<TCPClient>(std::move(socket));
}

void TCPClient::sendCommandForKey(CommandType type, const std::string& key) {
  int sock_fd = _socket.get_socket_fd();
  tcputil::sendValue<CommandType>(sock_fd, type);
  tcputil::sendString(sock_fd, key);
}

void sendBytes(const std::vector<std::uint8_t>& bytes) {
  int sock_fd = _socket.get_socket_fd();
  tcputil::sendVector<std::uint8_t>(sock_fd, bytes);
}

void TCPClient::sendStrings(std::vector<std::string> strings) {
  size_t size = strings.size();
  int sock_fd = _socket.get_socket_fd();
  tcputil::sendValue<size_t>(sock_fd, size);

  if (strings.empty()) {
    return;
  }

  for (auto& s : strings) {
    tcputil::sendString(sock_fd, s);
  }
}

}  // namespace detail

TCPStore::TCPStore(std::string host, uint16_t port = kDefaultPort,
                   bool is_master = false, size_t num_workers = 1,
                   std::chrono::milliseconds timeout = Store::kDefaultTimeout)
    : Store(timeout),
      _host(host),
      _port(port),
      _is_master(is_master),
      _num_workers(num_workers) {
  if (_is_master) {
    _server = detail::TCPServer::start();
  }

  _client = detail::TCPClient::connect(_host, _port, _timeout);

  waitWorkers();
}

void TCPStore::waitWorkers() {
  if (_num_workers == 0) {
    return;
  }

  add(_init_key, 1);
  auto begin = std::chrono::steady_clock::now();
  if (_server) {
    do {
      auto value = get(_init_key);
      int completed = std::strtoi(std::string(value.begin(), value.size()));
      if (completed >= _num_workers) {
        break;
      }
      const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - start);
      if (_timeout != kNoTimeout && elapsed > _timeout) {
        break;
      }
    } while (true);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void TCPStore::set(const std::string& key, const std::vector<uint8_t>& value) {
  const std::lock_guard<std::mutex> lock(_lock);
  _client->sendCommand(SET, _key_prefix + key);
  _client->sendBytes(value);
}

std::vector<uint8_t> TCPStore::get(const std::string& key) {
  const std::lock_guard<std::mutex> lock(_lock);
  doWait(key, _timeout);
  _client->sendCommandForKey(GET, _key_prefix + key);
  return _client->receiveBytes();
}

int64_t TCPStore::add(const std::string& key, int64_t value) {
  const std::lock_guard<std::mutex> lock(_lock);
  _client->sendCommandForKey(ADD, _key_prefix + key);
  _client->sendValue<std::int64_t>(value);
  return _client->receive<std::int64_t>();
}

bool removeKey(const std::string& key) override {
  const std::lock_guard<std::mutex> lock(_lock);
  _client->sendCommandForKey(REMOVE, _key_prefix + key);
  auto num_removed = _client->receiveValue<std::int64_t>();
  return num_removed == 1;
}
void wait(const std::vector<std::string>& keys) { wait(keys, timeout_); }

void TCPStore::wait(const std::vector<std::string>& keys,
                    const std::chrono::milliseconds& timeout) {
  std::vector<std::string> keys_with_prefix;
  for (auto key : keys) {
    keys_with_prefix.emplace_back(_key_prefix_ + key);
  }
  doWait(keys_with_prefix, timeout);
}

void TCPStore::doWait(const std::vector<std::string>& keys,
                      const std::chrono::milliseconds& timeout) {
  pre_timeout = _client->getTimeout();
  _client->sendCommand(WAIT);
  _client_->sendStrings(keys);

  _client->setTimeout(timeout);
  auto reply = client_->receive<WaitRelayType>();
  PADDLE_ENFORCE_EQ(replay, STOP_WAITING,
                    platform::errors::InvalidArgument(
                        "The call doWait must reply with STOP_WAITING."));
  _client->setTimeout(pre_timeout);
}

}  // namespace distributed
}  // namespace paddle
