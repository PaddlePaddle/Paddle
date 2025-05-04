// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/memory/circlable_ref_list.h"

namespace ap::memory {

class Guard final {
 public:
  Guard(const Guard&) = delete;
  Guard(Guard&&) = delete;
  Guard() : circlable_ref_list_(std::make_shared<CirclableRefList>()) {}

  const std::shared_ptr<CirclableRefListBase>& circlable_ref_list() const {
    return circlable_ref_list_;
  }

 private:
  std::shared_ptr<CirclableRefListBase> circlable_ref_list_;
};

}  // namespace ap::memory
