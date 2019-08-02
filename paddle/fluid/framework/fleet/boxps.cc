// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/fleet/boxps.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace boxps {
int FakeBoxPS::PassBegin(const std::set<uint64_t> &pass_data) {
  printf("FakeBoxPS: Pass begin...\n");
  for (const auto fea : pass_data) {
    if (emb_.find(fea) == emb_.end()) {
      emb_[fea] = std::vector<float>(hidden_size_, 0.0);
    }
  }

  for (auto i = emb_.begin(); i != emb_.end(); ++i) {
    printf("%lu: ", i->first);
    for (const auto e : i->second) {
      printf("%f ", e);
    }
    printf("\n");
  }
  return 0;
}

int FakeBoxPS::PassEnd() {
  printf("FakeBoxPS: Pass end...\n");
  return 0;
}

int FakeBoxPS::PullSparse(const std::vector<std::vector<uint64_t>> &keys,
                          std::vector<std::vector<float>> *values) {
  printf("FakeBoxPS:begin pull sparse...\n");
  auto slot_size = keys.size();
  for (auto slot_id = 0; slot_id < slot_size; ++slot_id) {
    for (auto i = 0; i < keys[slot_id].size(); ++i) {
      const auto iter = emb_.find(keys[slot_id][i]);
      if (iter == emb_.end()) {
        printf("Pull Sparse error - no key for %lu\n", keys[slot_id][i]);
        return 1;
      }
      const auto *value = iter->second.data();
      // memory in values has been allocated in pull_sparse_op
      memcpy(values->at(slot_id).data() + i * hidden_size_, value,
             hidden_size_ * sizeof(float));
    }
  }
  return 0;
}

int FakeBoxPS::PushSparse(const std::vector<std::vector<uint64_t>> &keys,
                          const std::vector<std::vector<float>> &values) {
  printf("FakeBoxPS:begin push grad sparse...\n");
  auto slot_size = keys.size();
  for (auto slot_id = 0; slot_id < slot_size; ++slot_id) {
    for (auto i = 0; i < keys[slot_id].size(); ++i) {
      auto iter = emb_.find(keys[slot_id][i]);
      if (iter == emb_.end()) {
        printf("Push Sparse grad error - no key for %lu\n", keys[slot_id][i]);
        return 1;
      }
      auto &para = iter->second;
      for (int j = 0; j < hidden_size_; ++j) {
        para[j] -= learning_rate_ * values[slot_id][i * hidden_size_ + j];
      }
    }
  }
  return 0;
}
}  // end namespace boxps
}  // end namespace paddle
