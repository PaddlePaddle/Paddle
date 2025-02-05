/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/pir/include/core/value.h"

namespace paddle {
namespace framework {

enum class ShapeMode {
  kMIN,
  kOPT,
  kMAX,
};

// CollectShapeManager can get all shape of value when run executor and this
// information will be used for TensorRTEngine
class CollectShapeManager {
 public:
  static CollectShapeManager& Instance();

  CollectShapeManager(const CollectShapeManager&) = delete;
  CollectShapeManager(CollectShapeManager&&) = delete;
  CollectShapeManager& operator=(const CollectShapeManager&) = delete;

  void SetValueMap(
      const std::unordered_map<pir::Value, pir::Value>& op_value2kernel_value) {
    op_value2kernel_value_ = op_value2kernel_value;
  }

  void CollectShapeInfo(framework::InstructionBase* instr,
                        framework::ValueExecutionInfo* value_exe_info,
                        framework::Scope* scope);
  void StatisticShapeRangeInfo();

  std::vector<int32_t> GetValueShapeRangeInfo(pir::Value val,
                                              bool is_shape_tensor,
                                              ShapeMode shape_mode);

  void ClearShapeInfo() {
    shape_info_.clear();
    shape_tensor_info_.clear();
    min_shapes_.clear();
    max_shapes_.clear();
    opt_shapes_.clear();
    min_values_.clear();
    max_values_.clear();
    opt_values_.clear();
    op_value2kernel_value_.clear();
    is_shape_range_info_ready_ = false;
  }

 private:
  CollectShapeManager() {}
  std::unordered_map<pir::Value, pir::Value> op_value2kernel_value_;
  std::map<pir::Value, std::vector<std::vector<int32_t>>> shape_info_;
  std::map<pir::Value, std::vector<std::vector<int32_t>>> shape_tensor_info_;
  std::map<pir::Value, std::vector<int32_t>> min_shapes_;
  std::map<pir::Value, std::vector<int32_t>> max_shapes_;
  std::map<pir::Value, std::vector<int32_t>> opt_shapes_;
  std::map<pir::Value, std::vector<int32_t>> min_values_;
  std::map<pir::Value, std::vector<int32_t>> max_values_;
  std::map<pir::Value, std::vector<int32_t>> opt_values_;
  bool is_shape_range_info_ready_ = false;
  std::mutex info_mutex_;
};

}  // namespace framework
}  // namespace paddle
