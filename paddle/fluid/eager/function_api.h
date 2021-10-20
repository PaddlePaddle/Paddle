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

#include "paddle/pten/hapi/all.h"

namespace egr {

// Public
void ScaleAPI(const paddle::experimental::Tensor& x, float scale, float bias,
              bool bias_after_scale, paddle::experimental::Tensor* out);
void FillConstAPI(double value, const ptenDDim& ddim,
                  const ptenBackend& backend, const ptenDataType& dtype,
                  const ptenDataLayout& layout,
                  paddle::experimental::Tensor* target);

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
  const int GetAMPLevel() const { return amp_level_; }

 private:
  Controller() = default;
  static Controller* controller_;
  std::shared_ptr<paddle::platform::Place> expected_place_ = nullptr;
  int amp_level_ = 0;
  DISABLE_COPY_AND_ASSIGN(Controller);
};

}  // namespace egr
