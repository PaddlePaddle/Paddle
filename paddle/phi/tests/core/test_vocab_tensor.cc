/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "gtest/gtest.h"

#include "paddle/phi/core/vocab_tensor.h"

namespace phi {
namespace tests {

TEST(VocabTensor, construct) {
  std::unordered_map<int32_t, std::int32_t> vocab = {{1, 1}, {2, 2}};

  auto check_vocab_tensor = [](VocabTensor& t,
                               std::unordered_map<int32_t, std::int32_t>& m) {
    bool r{true};
    r = r && (t.numel() == static_cast<int64_t>(m.size()));
    r = r && (t.place() == phi::CPUPlace());
    r = r && t.initialized();

    auto map = t.data();
    for (auto iter = m.begin(); iter != m.end(); iter++) {
      if (map.count(iter->first) == 0) {
        r = false;
        break;
      }
      r = r && (map[iter->first] == iter->second);
      if (!r) {
        return r;
      }
    }
    return r;
  };
  VocabTensor data(vocab);
  check_vocab_tensor(data, vocab);

  VocabTensor t_1(data);
  VocabTensor t_2 = data;
  VocabTensor t_3;
  t_3 = data;
  t_3 = std::move(data);
}

}  // namespace tests
}  // namespace phi
