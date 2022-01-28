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
#include "paddle/fluid/distributed/store/store.h"

namespace paddle {
namespace distributed {

class TCPStore : public Store {
 public:
  static constexpr std::uint16_t kDefaultPort = 6170;
  explicit TCPStore(std::string host, uint16_t port = kDefaultPort,
                    bool is_master = false, size_t num_workers = 1,
                    std::chrono::milliseconds timeout = Store::kDefaultTimeout);

  ~TCPStore() = default;

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

  const std::string _init_key = "init/";
  const std::string _key_prefix = "/";
  std::string _host;
  uint16_t _port;
  size_t _num_workers;
  std::mutex _lock;
}

}  // namespace distributed
}  // namespace paddle
