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

#include "paddle/fluid/distributed/store/store.h"
#include "paddle/fluid/distributed/store/tcp_utils.h"

namespace paddle {
namespace distributed {

enum class ReplyType { WAITING, STOP_WAIT };
enum class Command { ADD, GET, WAIT, STOP };

namespace detail {

class MasterDaemon {
 public:
  static std::unique_ptr<MasterDaemon> start(SocketType listen_socket);
  MasterDaemon() = delete;
  explicit MasterDaemon(SocketType listen_socket);
  ~MasterDaemon();

 private:
  void run();
  void _do_add(SocketType socket);
  void _do_wait(SocketType socket);
  void _do_get(SocketType socket);
  void _do_stop(SocketType socket);
  SocketType _listen_socket;
  std::vector<SocketType> _sockets;
  std::unordered_map<std::string, std::vector<uint8_t>> _store;
  std::thread _background_thread{};
  bool _stop = false;
};

class TCPServer {
 public:
  TCPServer() = default;
  static std::unique_ptr<TCPServer> create(std::uint16_t port);

 private:
  std::unique_ptr<MasterDaemon> _master_daemon;
};

class TCPClient {
 public:
  explicit TCPClient(SocketType socket) : _socket{socket} {}
  static std::unique_ptr<TCPClient> connect(const std::string host,
                                            uint16_t port);
  ~TCPClient() { tcputils::close_socket(_socket); }
  void send_command_for_key(Command type, const std::string& key);

  template <typename T>
  void send_value(const T& value);

  template <typename T>
  void send_vector(const std::vector<T>& value);
  template <typename T>
  std::vector<T> receive_vector();

  template <typename T>
  T receive_value();

 private:
  SocketType _socket;
};

}  // namespace detail

class TCPStore : public Store {
 public:
  static constexpr std::uint16_t kDefaultPort = 6170;
  explicit TCPStore(std::string host, uint16_t port = kDefaultPort,
                    bool is_master = false, size_t num_workers = 1,
                    std::chrono::seconds timeout = tcputils::kDefaultTimeout);

  ~TCPStore();

  int64_t add(const std::string& key, int64_t value) override;
  std::vector<uint8_t> get(const std::string& key) override;
  void wait(const std::string& key) override;

 private:
  void waitWorkers();
  std::unique_ptr<detail::TCPServer> _server;
  std::unique_ptr<detail::TCPClient> _client;

  const std::string _init_key = "init/";
  const std::string _key_prefix = "/";
  std::chrono::seconds _timeout;
  bool _is_master;
  int _num_workers;
};

}  // namespace distributed
}  // namespace paddle
