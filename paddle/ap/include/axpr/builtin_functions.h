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
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/core_expr.h"
#include "paddle/ap/include/axpr/value.h"

namespace ap::axpr {

Result<axpr::Value> BuiltinIdentity(const axpr::Value&,
                                    const std::vector<axpr::Value>& args);

Result<axpr::Value> BuiltinNot(const axpr::Value&,
                               const std::vector<axpr::Value>& args);

Result<axpr::Value> Raise(const axpr::Value&,
                          const std::vector<axpr::Value>& args);

Result<axpr::Value> BuiltinList(const axpr::Value&,
                                const std::vector<axpr::Value>& args);

Result<axpr::Value> BuiltinHalt(const axpr::Value&,
                                const std::vector<axpr::Value>& args);

adt::Result<axpr::Value> Print(InterpreterBase<axpr::Value>* interpreter,
                               const axpr::Value&,
                               const std::vector<axpr::Value>& args);

adt::Result<axpr::Value> ReplaceOrTrimLeftComma(
    const axpr::Value&, const std::vector<axpr::Value>& args);

adt::Result<axpr::Value> MakeRange(const axpr::Value&,
                                   const std::vector<axpr::Value>& args);

Result<axpr::Value> FlatMap(axpr::InterpreterBase<axpr::Value>* interpreter,
                            const axpr::Value&,
                            const std::vector<axpr::Value>& args);

Result<axpr::Value> ForEach(axpr::InterpreterBase<axpr::Value>* interpreter,
                            const axpr::Value&,
                            const std::vector<axpr::Value>& args);

Result<axpr::Value> Map(axpr::InterpreterBase<axpr::Value>* interpreter,
                        const axpr::Value&,
                        const std::vector<axpr::Value>& args);

Result<axpr::Value> Apply(axpr::InterpreterBase<axpr::Value>* interpreter,
                          const axpr::Value&,
                          const std::vector<axpr::Value>& args);

Result<axpr::Value> Length(axpr::InterpreterBase<axpr::Value>* interpreter,
                           const axpr::Value&,
                           const std::vector<axpr::Value>& args);

Result<axpr::Value> Filter(axpr::InterpreterBase<axpr::Value>* interpreter,
                           const axpr::Value&,
                           const std::vector<axpr::Value>& args);

Result<axpr::Value> Zip(const axpr::Value&,
                        const std::vector<axpr::Value>& args);

Result<axpr::Value> Reduce(axpr::InterpreterBase<axpr::Value>* interpreter,
                           const axpr::Value&,
                           const std::vector<axpr::Value>& args);

Result<axpr::Value> Max(const axpr::Value&,
                        const std::vector<axpr::Value>& args);

Result<axpr::Value> Min(const axpr::Value&,
                        const std::vector<axpr::Value>& args);

Result<axpr::Value> Min(const axpr::Value&,
                        const std::vector<axpr::Value>& args);

Result<axpr::Value> GetAttr(axpr::InterpreterBase<axpr::Value>* interpreter,
                            const axpr::Value&,
                            const std::vector<axpr::Value>& args);

Result<axpr::Value> SetAttr(axpr::InterpreterBase<axpr::Value>* interpreter,
                            const axpr::Value&,
                            const std::vector<axpr::Value>& args);
}  // namespace ap::axpr
