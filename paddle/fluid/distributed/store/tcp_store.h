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
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "paddle/fluid/distributed/store/socket.h"
#include "paddle/fluid/distributed/store/store.h"

namespace paddle {
namespace distributed {

enum class WaitReplyType { STOP_WAIT };

enum class Command { SET, GET, REMOVE, ADD, WAIT };

namespace detail {

class MasterDaemon {
 public:
  static std::unique_ptr<MasterDaemon> start(Socket sock);
  explicit MasterDaemon(Socket socket);
  ~MasterDaemon() {}

 private:
  void run();
  void doSet(int);
  void doGet(int);
  void doRemove(int);
  void doAdd(int);
  void doWait(int);
  Socket _socket;
  std::vector<Socket> _sockets;
  std::unordered_map<std::string, std::vector<uint8_t>> _store;
  std::thread _background_thread{};
};

class TCPServer {
 public:
  explicit TCPServer(uint16_t port) : _port(port) {}
  static std::unique_ptr<TCPServer> create(std::uint16_t port);

 private:
  std::uint16_t _port;
  std::unique_ptr<MasterDaemon> _master_daemon;
};

class TCPClient {
 public:
  // TCPClient(Socket&& socket) : _socket{std::move(socket)} {}
  explicit TCPClient(Socket socket) : _socket{socket} {}
  static std::unique_ptr<TCPClient> connect(
      const std::string host, uint16_t port,
      const std::chrono::milliseconds& timeout);
  void sendCommandForKey(Command type, const std::string& key);
  void sendCommand(Command type);
  void sendStrings(std::vector<std::string> strings);
  void sendBytes(const std::vector<std::uint8_t>& bytes);
  std::vector<uint8_t> recvBytes();

  template <typename T>
  void sendValue(const T& value);

  template <typename T>
  T recvValue();

 private:
  Socket _socket;
};

}  // namespace detail

class TCPStore : public Store {
 public:
  static constexpr std::uint16_t kDefaultPort = 6170;
  explicit TCPStore(std::string host, uint16_t port = kDefaultPort,
                    bool is_master = false, size_t num_workers = 1,
                    std::chrono::milliseconds timeout = Store::kDefaultTimeout);

  ~TCPStore() = default;
  void setTimeout(std::chrono::milliseconds timeout) { _timeout = timeout; }

  void set(const std::string& key, const std::vector<uint8_t>& value) override;
  std::vector<uint8_t> get(const std::string& key) override;

  bool removeKey(const std::string& key) override;
  int64_t add(const std::string& key, int64_t value) override;
  void wait(const std::vector<std::string>& keys) override;
  void wait(const std::vector<std::string>& keys,
            const std::chrono::milliseconds& timeout) override;

 private:
  void doWait(const std::vector<std::string>& keys,
              const std::chrono::milliseconds& timeout);
  void waitWorkers();
  std::unique_ptr<detail::TCPServer> _server;
  std::unique_ptr<detail::TCPClient> _client;

  const std::string _init_key = "init/";
  const std::string _key_prefix = "/";
  std::chrono::milliseconds _timeout;
  std::string _host;
  uint16_t _port;
  bool _is_master;
  int _num_workers;
  std::mutex _lock;
};

}  // namespace distributed
}  // namespace paddle
