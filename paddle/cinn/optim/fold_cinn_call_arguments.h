// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <string>

#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 * \brief Rewrite the Call Nodes marked type as CINN, pack their arguments into
 * `void*, int` so that they can trigger a `LoweredFunc`.
 *
 * For example, input the IR
 * \code
 * Call(some_lowered_func, a:cinn_buffer_t*, b:cinn_buffer_t*, c:cinn_buffer_t*)
 * \endcode
 *
 * This pass will rewrite it to
 * \code
 * cinn_pod_value_t a_(a);
 * cinn_pod_value_t b_(b);
 * cinn_pod_value_t c_(c);
 *
 * cinn_args_construct(packed_args, a_, b_, c_);
 * Call(some_lowered_func, packed_args, 3); // 3 is the number of arguments
 * \endcode
 */
void FoldCINNCallArguments(ir::LoweredFunc fn);

}  // namespace optim
}  // namespace cinn
