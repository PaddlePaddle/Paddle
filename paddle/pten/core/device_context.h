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

#pragma once

// TODO(wilber): need to replace fluid place.
#include "paddle/fluid/platform/place.h"

namespace pten {

class Allocator;

using Place = paddle::platform::Place;

// TODO(wilber): should be a pure virtual class?
class DeviceContext {
 public:
  void SetAllocator(Allocator* allocator) noexcept { allocator_ = allocator; }
  Allocator* GetAllocator() const noexcept { return allocator_; }

  // TODO(wilber): code_style?
  virtual Place GetPlace() const noexcept = 0;

  virtual ~DeviceContext() {}

  virtual void Wait() {}

 protected:
  Allocator* allocator_{nullptr};
};

}  // namespace pten
