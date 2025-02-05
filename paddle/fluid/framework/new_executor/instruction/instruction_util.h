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

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/phi/api/profiler/event.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"
namespace paddle {
namespace framework {

class ValueExecutionInfo;

std::vector<int> GetValueIds(pir::Value value,
                             const ValueExecutionInfo& value_exec_info);

phi::DeviceContext* ParseDeviceContext(pir::Operation* op,
                                       phi::DeviceContext* origin_dev_ctx,
                                       const phi::Place& place,
                                       const std::string& execution_stream,
                                       const int stream_priority);

OpFuncType AnalyseOpFuncType(::pir::Operation* op, const phi::Place& place);

void GetInputIds(pir::Operation* op,
                 const ValueExecutionInfo& value_exec_info,
                 std::unordered_map<pir::Value, std::vector<int>>* input_ids);

std::vector<pir::Value> GetExternalInputs(
    pir::Block* block,
    const ValueExecutionInfo& value_exec_info,
    std::unordered_map<pir::Value, std::vector<int>>* input_ids);

void InsertTuplePushContainerToOuts(
    pir::Block* block,
    const ValueExecutionInfo& value_exec_info,
    std::unordered_map<pir::Value, std::vector<int>>* outputs);

void InsertInplacedExternalInputsToOuts(
    pir::Block* block,
    const std::vector<pir::Value>& external_inputs,
    const ValueExecutionInfo& value_exec_info,
    std::unordered_map<pir::Value, std::vector<int>>* outputs);

bool GetCondData(const phi::DenseTensor& cond);

void HandleForInplaceOp(pir::Operation* op,
                        const ValueExecutionInfo* value_exe_info,
                        InstructionBase* instr);

void ShareVarBuffer(const Variable* src_var, Variable* dst_var);
}  // namespace framework
}  // namespace paddle
