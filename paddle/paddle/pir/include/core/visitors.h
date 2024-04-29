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

#include <functional>

#include "paddle/pir/include/core/dll_decl.h"

namespace pir {

class Operation;
class Region;
class Block;

// Traversal order.
enum class WalkOrder { PreOrder, PostOrder };

// Defines utilities for walking and visiting operations.
IR_API void Walk(Operation *op,
                 const std::function<void(Region *)> &callback,
                 WalkOrder order);

IR_API void Walk(Operation *op,
                 const std::function<void(Block *)> &callback,
                 WalkOrder order);

IR_API void Walk(Operation *op,
                 const std::function<void(Operation *)> &callback,
                 WalkOrder order);

template <WalkOrder Order = WalkOrder::PostOrder, typename FuncTy>
void Walk(Operation *op, FuncTy &&callback) {
  return Walk(op, callback, Order);
}

}  // namespace pir
