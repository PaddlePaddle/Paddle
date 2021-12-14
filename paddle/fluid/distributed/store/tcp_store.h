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

#include <store.h>
#include <iostream>

namespace paddle {
namespace distributed {

class TCPStore : public Store {
 public:
  explicit TCPStore(std::string host);

  ~TCPStore() = default;

  void set(const std::string& key, const std::vector<uint8_t>& value) override;
  std::vector<uint8_t> get(const std::string& key) override;

  int64_t add(const std::string& key, int64_t value) override;
  bool removeKey(const std::string& key) override;
  void wait(const std::vector<std::string>& keys) override;
  void wait(const std::vector<std::string>& keys,
            const std::chrono::milliseconds& timeout) override;

  const std::string& getHost() const { return addr_.host; }

  std::uint16_t getPort() const { return addr_.port; }

 private:
  std::vector<uint8_t> doGet(const std::string& key);
  void doWait(const std::vector<std::string>& keys,
              const std::chrono::milliseconds& timeout);
  void waitWorkers();

  const std::string initKey_ = "init/";
  const std::string keyPrefix_ = "/";
  std::mutex lock_;
}

}  // namespace distributed
}  // namespace paddle
