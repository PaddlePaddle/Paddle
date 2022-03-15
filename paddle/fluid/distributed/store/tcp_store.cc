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

std::unique_ptr<MasterDaemon> MasterDaemon::start(SocketType socket,
                                                  int nranks) {
  return std::make_unique<MasterDaemon>(socket, nranks);
}

MasterDaemon::MasterDaemon(SocketType socket, int nranks)
    : _listen_socket(socket), _nranks(nranks) {
  _background_thread = std::thread{&MasterDaemon::run, this};
}

MasterDaemon::~MasterDaemon() {
  _background_thread.join();
  tcputils::close_socket(_listen_socket);
  for (SocketType socket : _sockets) {
    tcputils::close_socket(socket);
  }
}

void MasterDaemon::_do_add(SocketType socket) {
  int64_t new_value{};
  std::string key = tcputils::receive_string(socket);
  new_value = tcputils::receive_value<int64_t>(socket);
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
  VLOG(3) << "TCPStore: new value (" << new_value << ") for key (" << key
          << ").";
  tcputils::send_value<int64_t>(socket, new_value);
}

void MasterDaemon::_do_set(SocketType socket) {
  VLOG(3) << "MasterDaemon::_do_set";
  std::string key = tcputils::receive_string(socket);
  auto value = tcputils::receive_vector<uint8_t>(socket);
  _store[key] = value;
}

void MasterDaemon::_do_get(SocketType socket) {
  VLOG(3) << "MasterDaemon::_do_get";
  std::string key = tcputils::receive_string(socket);
  auto iter = _store.find(key);
  PADDLE_ENFORCE_NE(
      iter, _store.end(),
      platform::errors::InvalidArgument("Key %s not found in TCPStore.", key));
  std::vector<uint8_t> value = iter->second;
  tcputils::send_vector<uint8_t>(socket, value);
}

void MasterDaemon::_do_stop(SocketType socket) {
  VLOG(3) << "MasterDaemon::_do_stop";
  ReplyType value = ReplyType::STOP_WAIT;
  tcputils::send_value<ReplyType>(socket, value);
  if (--_nranks == 0) {
    _stop = true;
  }
}

void MasterDaemon::_do_wait(SocketType socket) {
  VLOG(3) << "MasterDaemon::_do_wait";
  std::string key = tcputils::receive_string(socket);
  auto iter = _store.find(key);
  auto reply = ReplyType::STOP_WAIT;
  if (iter == _store.end()) {
    reply = ReplyType::WAITING;
  }
  VLOG(3) << "TCPStore: wait reply (" << static_cast<int>(reply)
          << ") for key (" << key << ").";
  tcputils::send_value<ReplyType>(socket, reply);
}

void MasterDaemon::run() {
  std::vector<struct pollfd> fds;
#ifdef _WIN32
  fds.push_back({_listen_socket, POLLIN});
#else
  fds.push_back({.fd = _listen_socket, .events = POLLIN, .revents = 0});
#endif

  while (!_stop) {
    for (size_t i = 0; i < fds.size(); i++) {
      fds[i].revents = 0;
    }

#ifdef _WIN32
    ::WSAPoll(fds.data(), fds.size(), INFTIME);
#else
    ::poll(fds.data(), fds.size(), INFTIME);
#endif

    if (fds[0].revents != 0) {
      auto socket = tcputils::tcp_accept(_listen_socket);
      _sockets.emplace_back(socket);
#ifdef _WIN32
      fds.push_back({socket, POLLIN});
#else
      fds.push_back({.fd = socket, .events = POLLIN, .revents = 0});
#endif
    }

    for (size_t i = 1; i < fds.size(); i++) {
      try {
        if (fds[i].revents == 0) {
          continue;
        }

        Command command = tcputils::receive_value<Command>(fds[i].fd);
        VLOG(3) << "TCPStore: recv command: " << static_cast<int>(command)
                << ".";

        switch (command) {
          case Command::ADD:
            _do_add(fds[i].fd);
            break;
          case Command::GET:
            _do_get(fds[i].fd);
            break;
          case Command::SET:
            _do_set(fds[i].fd);
            break;
          case Command::WAIT:
            _do_wait(fds[i].fd);
            break;
          case Command::STOP:
            _do_stop(fds[i].fd);
            break;
          default:
            VLOG(0) << "Unknow command: " << static_cast<int>(command);
            exit(-1);
        }
      } catch (...) {
        fds.erase(fds.begin() + i);
        _sockets.erase(_sockets.begin() + i - 1);
      }
    }
  }
}

std::unique_ptr<TCPServer> TCPServer::create(uint16_t port, int nranks) {
  int socket = tcputils::tcp_listen("", std::to_string(port), AF_INET);
  auto server = std::make_unique<TCPServer>();
  server->_master_daemon = MasterDaemon::start(socket, nranks);
  return server;
}

std::unique_ptr<TCPClient> TCPClient::connect(const std::string host,
                                              uint16_t port) {
  int socket = tcputils::tcp_connect(host, std::to_string(port), AF_INET);
  return std::make_unique<TCPClient>(socket);
}

void TCPClient::send_command_for_key(Command type, const std::string& key) {
  tcputils::send_value<Command>(_socket, type);
  if (key.empty()) {
    return;
  }
  tcputils::send_string(_socket, key);
}

template <typename T>
void TCPClient::send_value(const T& value) {
  tcputils::send_bytes<T>(_socket, &value, 1);
}

template <typename T>
T TCPClient::receive_value() {
  T res;
  tcputils::receive_bytes<T>(_socket, &res, 1);
  return res;
}

template <typename T>
void TCPClient::send_vector(const std::vector<T>& value) {
  tcputils::send_vector<T>(_socket, value);
}

template <typename T>
std::vector<T> TCPClient::receive_vector() {
  return tcputils::receive_vector<T>(_socket);
}

}  // namespace detail

TCPStore::TCPStore(std::string host, uint16_t port, bool is_master,
                   size_t num_workers, std::chrono::seconds timeout)
    : Store(timeout), _is_master(is_master), _num_workers(num_workers) {
  if (_is_master) {
    _server = detail::TCPServer::create(port, num_workers);
  }

  _client = detail::TCPClient::connect(host, port);
  waitWorkers();
}

void TCPStore::waitWorkers() {
  if (_num_workers == 0) {
    return;
  }
  add(_init_key, 1);

  auto begin = std::chrono::steady_clock::now();
  do {
    auto value = get(_init_key);
    int completed = std::stoi(std::string(value.begin(), value.end()));
    VLOG(3) << completed << " worker ready, total " << _num_workers;
    if (completed >= _num_workers) {
      break;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - begin);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (_timeout != tcputils::kNoTimeout && elapsed > _timeout) {
      PADDLE_ENFORCE_EQ(
          completed, _num_workers,
          platform::errors::InvalidArgument(
              "TCPStore timeouted and not all workers got ready."));
    }
  } while (true);
  VLOG(3) << "TCPStore initialized.";
}

int64_t TCPStore::add(const std::string& key, int64_t value) {
  VLOG(3) << "TCPStore add.";
  _client->send_command_for_key(Command::ADD, _key_prefix + key);
  _client->send_value<std::int64_t>(value);
  return _client->receive_value<std::int64_t>();
}

void TCPStore::set(const std::string& key, const std::vector<uint8_t>& value) {
  VLOG(3) << "TCPStore set.";
  _client->send_command_for_key(Command::SET, _key_prefix + key);
  _client->send_vector<std::uint8_t>(value);
}

std::vector<uint8_t> TCPStore::get(const std::string& key) {
  wait(key);
  _client->send_command_for_key(Command::GET, _key_prefix + key);
  VLOG(3) << "TCPStore get.";
  return _client->receive_vector<uint8_t>();
}

void TCPStore::wait(const std::string& key) {
  ReplyType reply;
  VLOG(3) << "TCPStore wait.";
  do {
    _client->send_command_for_key(Command::WAIT, _key_prefix + key);

    reply = _client->receive_value<ReplyType>();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  } while (reply != ReplyType::STOP_WAIT);
}

TCPStore::~TCPStore() {
  VLOG(3) << "~TCPStore";
  _client->send_command_for_key(Command::STOP, "");
  ReplyType ret = _client->receive_value<ReplyType>();
  PADDLE_ENFORCE_EQ(ret, ReplyType::STOP_WAIT,
                    platform::errors::InvalidArgument(
                        "The reply for TCPStore destructure must be 0."));
}

}  // namespace distributed
}  // namespace paddle
