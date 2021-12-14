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

#include <tcp_store.h>
#include <iostream>

namespace paddle {
namespace distributed {

TCPStore::TCPStore(std::string host, const TCPStoreOptions& opts)
    : Store{opts.timeout},
      addr_{std::move(host)},
      num_workers_{opts.num_workers} {
  if (opts.is_server) {
    server_ = detail::TCPServer::start(opts);
    addr_.port = server_->port();
  } else {
    addr_.port = opts.port;
  }
  client_ = detail::TCPClient::connect(addr_, opts);

  waitWorkers();
}

void TCPStore::waitWorkers() {
  if (num_workers == 0) {
    return;
  }

  add(initKey, 1);
  if (server_) {
    do {
      const auto begin = std::chrono::steady_clock::now();
      std::vector<uint8_t> v = doGet(initKey_);
      int completed = std::stoi(std::string(v.data(), v.size()));
      if (completed >= num_workers_) {
        break;
      }
      const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - start);
      if (timeout_ != kNoTimeout && elapsed > timeout_) {
        break;
      }
    } while (true);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void TCPStore::set(const std::string& key, const std::vector<uint8_t>& value) {
  client_->sendCommand(SET, keyPrefix_ + key);
  client_->sendBytes(value);
}

std::vector<uint8_t> TCPStore::get(const std::string& key) {
  return doGet(keyPrefix_ + key);
}

int64_t TCPStore::add(const std::string& key, int64_t value) {
  client_->sendCommand(ADD, key);
  client_->sendValue<std::int64_t>(value);
  return client_->receive<std::int64_t>();
}
bool removeKey(const std::string& key) override;
void wait(const std::vector<std::string>& keys) { wait(keys, timeout_); }

void TCPStore::wait(const std::vector<std::string>& keys,
                    const std::chrono::milliseconds& timeout) {
  std::vector<std::string> keys_with_prefix;
  for (auto key : keys) {
    keys_with_prefix.emplace_back(keyPrefix_ + key);
  }
  doWait(keys_with_prefix, timeout);
}

std::vector<uint8_t> TCPStore::doGet(const std::string& key) {
  doWait(key, timeout_);
  client_->sendCommand(GET, key);
  return client_->receiveBytes();
}

void TCPStore::doWait(const std::vector<std::string>& keys,
                      const std::chrono::milliseconds& timeout) {
  pre_timeout = client_->getTimeout();
  client_->sendCommand(WAIT);
  client_->sendStrings(keys);

  auto reply = client_->receive<WaitRelayType>();
  if (reply != STOP_WAITING) {
    client_->setTimeout(pre_timeout);
    throw something;
  }
  client_->setTimeout(pre_timeout);
}

}  // namespace distributed
}  // namespace paddle
