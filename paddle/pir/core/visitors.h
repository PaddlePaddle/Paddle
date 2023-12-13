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
#include "paddle/pir/core/dll_decl.h"
#include "paddle/pir/core/region.h"

namespace pir {

class Operation;

/// Traversal order.
enum class WalkOrder { PreOrder, PostOrder };

namespace detail {
// template <typename Iterator>
// /// Defines utilities for walking and visiting operations.
// IR_API void Walk(Operation *op,
//                  std::function<void(Region *)> callback,
//                  WalkOrder order);

// template <typename Iterator>
// IR_API void Walk(Operation *op,
//                  std::function<void(Block *)> callback,
//                  WalkOrder order);

// template <typename Iterator>
// IR_API void Walk(Operation *op,
//                  std::function<void(Operation *)> callback,
//                  WalkOrder order);

// template <WalkOrder Order = WalkOrder::PostOrder,
//           typename Iterator,  // = ForwardIterator,
//           typename FuncTy>
// IR_API void Walk(Operation *op, FuncTy &&callback);

}  // namespace detail
}  // namespace pir
