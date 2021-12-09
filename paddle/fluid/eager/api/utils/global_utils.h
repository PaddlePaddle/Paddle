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
//

#pragma once

#include <atomic>
#include <memory>
#include "paddle/fluid/platform/place.h"

namespace egr {

class UniqueNameGenerator {
 public:
  explicit UniqueNameGenerator(std::string prefix = "") : prefix_(prefix) {}
  std::string Generate(std::string key = "eager_tmp") {
    return prefix_ + key + "_" + std::to_string(id_++);
  }

 private:
  std::atomic<int> id_{0};
  std::string prefix_;
};

// Global
class Controller {
 public:
  static Controller& Instance() { return *controller_; }
  const paddle::platform::Place& GetExpectedPlace() const {
    return *expected_place_.get();
  }
  void SetExpectedPlace(const paddle::platform::Place& place) {
    expected_place_ = std::make_shared<paddle::platform::Place>(place);
  }
  void SetAMPLevel(int level) { amp_level_ = level; }
  int GetAMPLevel() const { return amp_level_; }
  bool HasGrad() const { return has_grad_; }
  std::string GenerateUniqueName(std::string key = "eager_tmp") {
    return generator_->Generate(key);
  }

 private:
  Controller() = default;
  static Controller* controller_;
  std::shared_ptr<paddle::platform::Place> expected_place_ = nullptr;
  int amp_level_ = 0;
  bool has_grad_ = true;
  std::unique_ptr<UniqueNameGenerator> generator_{new UniqueNameGenerator()};
  DISABLE_COPY_AND_ASSIGN(Controller);
};

}  // namespace egr
