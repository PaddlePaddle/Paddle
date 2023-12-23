// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pir/core/operation.h"

namespace pir {
namespace drr {

class IrOperation {
 public:
  explicit IrOperation(pir::Operation* op) : op_(op) {}

  pir::Operation* get() const { return op_; }

 private:
  pir::Operation* op_;
};

}  // namespace drr
}  // namespace pir
