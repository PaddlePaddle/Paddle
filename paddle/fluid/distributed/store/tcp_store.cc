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

#include <chrono>
#include <iostream>
#include <thread>

#include "paddle/fluid/distributed/store/tcp_store.h"
#include "paddle/fluid/distributed/store/tcp_utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

namespace detail {

std::unique_ptr<TCPServer> TCPServer::create(uint16_t port) {
  // auto socket = Socket::listen(port);
  // return std::make_unique<TCPServer>(std::move(socket));
  return std::make_unique<TCPServer>(port);
}

std::unique_ptr<TCPClient> TCPClient::connect(
    const std::string host, uint16_t port,
    const std::chrono::milliseconds& timeout) {
  SocketOptions sock_opt{};
  sock_opt.set_connect_timeout(timeout);
  const Socket socket = Socket::connect(host, port, sock_opt);
  // return std::make_unique<TCPClient>(std::move(socket));
  return std::make_unique<TCPClient>(socket);
}

void TCPClient::sendCommandForKey(Command type, const std::string& key) {
  int sock_fd = _socket.get_socket_fd();
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

  add(_init_key, 1);
  auto begin = std::chrono::steady_clock::now();
  if (_server) {
    do {
      auto value = get(_init_key);
      int completed = std::stoi(std::string(value.begin(), value.end()));
      if (completed >= _num_workers) {
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
  const std::lock_guard<std::mutex> lock(_lock);
  _client->sendCommandForKey(Command::ADD, _key_prefix + key);
  _client->sendValue<std::int64_t>(value);
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
