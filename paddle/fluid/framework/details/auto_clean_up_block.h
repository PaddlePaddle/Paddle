// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <functional>
namespace paddle {
namespace framework {
namespace details {

class AutoCleanUpBlock {
 public:
  explicit AutoCleanUpBlock(std::function<void()>&& clean_up)
      : clean_up_(std::move(clean_up)) {}

  template <typename InitCallback>
  AutoCleanUpBlock(InitCallback init, std::function<void()>&& clean_up)
      : clean_up_(std::move(clean_up)) {
    init();
  }

  ~AutoCleanUpBlock() {
    if (clean_up_) {
      clean_up_();
    }
  }

  AutoCleanUpBlock(const AutoCleanUpBlock&) = delete;
  AutoCleanUpBlock& operator=(const AutoCleanUpBlock&) = delete;

  AutoCleanUpBlock(AutoCleanUpBlock&& block) {
    clean_up_ = block.clean_up_;
    block.clean_up_ = [] {};
  }

  AutoCleanUpBlock& operator=(AutoCleanUpBlock&& block) {
    clean_up_ = block.clean_up_;
    block.clean_up_ = [] {};
    return *this;
  }

 private:
  std::function<void()> clean_up_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
