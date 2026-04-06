// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// The file has been adapted from pytorch project
// Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <ATen/core/function_schema.h>
#include <c10/macros/Macros.h>
#include <string>
#include <variant>

namespace torch::jit {

// allow_typevars: If true, we assume that lowercase types that we don't
// understand are type variables. This is only needed for TorchScript (and not
// not needed for custom ops).
// If false, we disallow typevars, except in certain cases for BC reason (i.e.
// your op is in the aten or prim namespace).
PADDLE_API std::variant<std::string, c10::FunctionSchema> parseSchemaOrName(
    const std::string& schemaOrName);
PADDLE_API c10::FunctionSchema parseSchema(const std::string& schema);
PADDLE_API std::string parseName(const std::string& name);
PADDLE_API std::string schemaTypeTreeToDebugString(
    const c10::FunctionSchema& schema);
}  // namespace torch::jit
