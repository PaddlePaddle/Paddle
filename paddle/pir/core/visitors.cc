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

#include "paddle/pir/core/visitors.h"

namespace pir::detail {

/// Defines utilities for walking and visiting operations.
template <typename Iterator>
void Walk(Operation *op,
          std::function<void(Region *)> callback,
          WalkOrder order) {
  // No early increment here for they can't be erased from a callback.
  for (auto &region : Iterator::makeIterable(*op)) {
    if (order == WalkOrder::PreOrder) callback(&region);
    for (auto &block : Iterator::makeIterable(region)) {
      for (auto &nestedOp : Iterator::makeIterable(block))
        Walk<Iterator>(&nestedOp, callback, order);
    }
    if (order == WalkOrder::PostOrder) callback(&region);
  }
}

template <typename Iterator>
void Walk(Operation *op,
          std::function<void(Block *)> callback,
          WalkOrder order) {
  for (auto &region : Iterator::makeIterable(*op)) {
    // Early increment here in the case where the block is erased.
    for (auto &block :
         llvm::make_early_inc_range(Iterator::makeIterable(region))) {
      if (order == WalkOrder::PreOrder) callback(&block);
      for (auto &nestedOp : Iterator::makeIterable(block))
        Walk<Iterator>(&nestedOp, callback, order);
      if (order == WalkOrder::PostOrder) callback(&block);
    }
  }
}

template <typename Iterator>
void Walk(Operation *op,
          std::function<void(Operation *)> callback,
          WalkOrder order) {
  if (order == WalkOrder::PreOrder) callback(op);

  // TODO(zhangbopd): This walk should be iterative over the operations.
  for (auto &region : Iterator::makeIterable(*op)) {
    for (auto &block : Iterator::makeIterable(region)) {
      // Early increment here in the case where the operation is erased.
      for (auto &nestedOp :
           llvm::make_early_inc_range(Iterator::makeIterable(block)))
        Walk<Iterator>(&nestedOp, callback, order);
    }
  }

  if (order == WalkOrder::PostOrder) callback(op);
}

template <WalkOrder Order = WalkOrder::PostOrder,
          typename Iterator,  // = ForwardIterator,
          typename FuncTy>
IR_API void Walk(Operation *op, FuncTy &&callback) {
  return detail::walk<Iterator>(op, callback, Order);
}

}  // namespace pir::detail
