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

#include "paddle/pir/include/core/visitors.h"
#include "paddle/pir/include/core/operation.h"

namespace pir {

// Defines utilities for walking and visiting operations.
void Walk(Operation *op,
          const std::function<void(Region *)> &callback,
          WalkOrder order) {
  // No early increment here for they can't be erased from a callback.
  for (auto &region : *op) {
    if (order == WalkOrder::PreOrder) callback(&region);
    for (auto &block : region) {
      for (auto &op_item : block) {
        Walk(&op_item, callback, order);
      }
    }
    if (order == WalkOrder::PostOrder) callback(&region);
  }
}

void Walk(Operation *op,
          const std::function<void(Block *)> &callback,
          WalkOrder order) {
  for (auto &region : *op) {
    // Early increment here in the case where the block is erased.
    for (auto &block : region) {
      if (order == WalkOrder::PreOrder) callback(&block);

      for (auto &op_item : block) {
        Walk(&op_item, callback, order);
      }

      if (order == WalkOrder::PostOrder) callback(&block);
    }
  }
}

void Walk(Operation *op,
          const std::function<void(Operation *)> &callback,
          WalkOrder order) {
  if (order == WalkOrder::PreOrder) callback(op);

  // TODO(zhangbopd): This walk should be iterative over the operations.
  for (auto &region : *op) {
    for (auto &block : region) {
      // Early increment here in the case where the operation is erased.
      for (auto &op_item : block) Walk(&op_item, callback, order);
    }
  }

  if (order == WalkOrder::PostOrder) callback(op);
}

}  // namespace pir
