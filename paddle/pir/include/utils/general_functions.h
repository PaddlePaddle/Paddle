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

#include <string>
#include <vector>

#include "paddle/pir/include/core/type.h"

namespace pir {

class Operation;
class Block;
class Value;

/**
* @brief Get the value of the input and output of the specified op in the
external block.
*
* @param const Operation& const reference to an operation

* @return std::vector<Value>
*/
std::vector<Value> GetUsedExternalValuePir(const Operation& op);

/**
 * @brief Get the external value of the input and output of all op which in the
 specified block.
 *
 * @param const Block& const reference to an block

 * @return std::vector<Value>
 */
std::vector<Value> GetUsedExternalValuePir(const Block& block);

}  // namespace pir
