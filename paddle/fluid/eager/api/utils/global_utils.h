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
#include "paddle/fluid/imperative/tracer.h"

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
// TODO(jiabin): Now we are using imperative tracer, move it here when we
// deprecate imperative.

class Controller {
 public:
  static Controller& Instance() { return *controller_; }
  paddle::platform::Place GetExpectedPlace() const {
    return tracer_->ExpectedPlace();
  }
  void SetExpectedPlace(const paddle::platform::Place& place) {
    tracer_->SetExpectedPlace(place);
  }
  void SetAMPLevel(paddle::imperative::AmpLevel level) {
    tracer_->SetAmpLevel(level);
  }
  paddle::imperative::AmpLevel GetAMPLevel() const {
    return tracer_->GetAmpLevel();
  }
  bool HasGrad() const { return tracer_->HasGrad(); }
  void SetHasGrad(bool has_grad) { tracer_->SetHasGrad(has_grad); }
  std::string GenerateUniqueName(std::string key = "eager_tmp") {
    return tracer_->GenerateUniqueName(key);
  }
  const std::shared_ptr<paddle::imperative::Tracer>& GetCurrentTracer() {
    return tracer_;
  }
  void SetCurrentTracer(
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    tracer_ = tracer;
    VLOG(6) << "Set current tracer for Controller: " << tracer_;
  }

  bool InEagerMode() const { return in_eager_mode_; }

  void SetInEagerMode(bool in_eager_mode) { in_eager_mode_ = in_eager_mode; }

 private:
  Controller() = default;
  static Controller* controller_;
  std::shared_ptr<paddle::imperative::Tracer> tracer_{
      new paddle::imperative::Tracer()};
  // TODO(jiabin): remove when we don't need imperative.
  bool in_eager_mode_{false};
  DISABLE_COPY_AND_ASSIGN(Controller);
};

}  // namespace egr
