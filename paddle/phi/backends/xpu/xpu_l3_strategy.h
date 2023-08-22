/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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
#include <algorithm>
#include <numeric>
#include <vector>

namespace phi {

struct XPUL3CacheBlock {
 public:
  void Clear() {
    addr_ = nullptr;
    size_ = 0;
    history_.clear();
  }
  void Set(void* addr, size_t size);
  void Record(size_t size) { history_.push_back(size); }
  void* data() { return addr_; }
  size_t size() { return size_; }

 private:
  void* addr_{nullptr};
  size_t size_{0};

 public:
  std::vector<size_t> history_;
};

class XPUL3Planner {
 public:
  void RunAutotune(const std::vector<XPUL3CacheBlock*>& l3_block_dict,
                   size_t l3_size);

  std::vector<size_t>* plan() { return &plan_; }

 private:
  std::vector<size_t> plan_;
};

}  // namespace phi
