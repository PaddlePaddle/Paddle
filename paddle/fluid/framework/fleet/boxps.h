/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

// Just a abstract class and a stub implementation now
// Fengchao will support the actual function
// Plan to be ready in the end of September

#include <map>
#include <set>
#include <vector>

namespace paddle {
namespace boxps {

class BoxPS {
 public:
  BoxPS() {}
  virtual ~BoxPS() {}

  virtual int init(int hidden_size) = 0;
  virtual int PassBegin(const std::set<uint64_t> &pass_data) = 0;
  virtual int PassEnd() = 0;
  virtual int PullSparse(const std::vector<std::vector<uint64_t>> &keys,
                         std::vector<std::vector<float>> *values) = 0;
  virtual int PushSparse(
      const std::vector<std::vector<uint64_t>> &keys,
      const std::vector<std::vector<float>> &grad_values) = 0;
};

class FakeBoxPS : public BoxPS {
 public:
  FakeBoxPS() {}
  virtual ~FakeBoxPS() {}

  int init(int hidden_size) override {
    hidden_size_ = hidden_size;
    return 0;
  }
  int PassBegin(const std::set<uint64_t> &pass_data) override;

  int PassEnd() override;

  int PullSparse(const std::vector<std::vector<uint64_t>> &keys,
                 std::vector<std::vector<float>> *values) override;
  int PushSparse(const std::vector<std::vector<uint64_t>> &keys,
                 const std::vector<std::vector<float>> &grad_values) override;

 private:
  std::map<uint64_t, std::vector<float>> emb_;
  int hidden_size_ = 1;
  float learning_rate_ = 0.01;
};
}  // namespace boxps
}  // namespace paddle
