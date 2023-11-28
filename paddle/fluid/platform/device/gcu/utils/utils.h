/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include <functional>
#include <map>
#include <memory>
#include <mutex>  // NOLINT [build/c++11]
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dtu/hlir/builder/hlir_builder.h"
#include "gcu/umd/dtu_assembler_def.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_executable.h"
#include "paddle/fluid/platform/device/gcu/utils/layout.h"
#include "paddle/fluid/platform/device/gcu/utils/types.h"
#include "paddle/phi/common/data_type.h"

namespace hlir {
class HlirDispatch;
}  // namespace hlir

namespace paddle {
namespace platform {
namespace gcu {
using ExecutablePtr = runtime::ExecutablePtr;
using DispatchPtr = std::shared_ptr<hlir::HlirDispatch>;

static std::set<std::string> kUnusedArchetype = {"ReserveSpace"};

class TransformUtil {
 public:
  static std::string GetShapeStr(const std::vector<int64_t>& shape);
  /*
   * Parameters:
   *     type: [DataType] the data type
   * Return：
   *     [GcuPrimitiveType] the data type for hlir tensor
   * */
  static GcuPrimitiveType ConvertDataType(const DataType& type);

  /*
   * Parameters:
   *     type: [DataType] the data type of GCU UMD
   * Return：
   *     [TypeId] the data type for ME tensor
   * */
  static DataType ConvertGcuDataType(const dtu_umd::DataType& type);

  /*
   * Parameters:
   *     gcu_builder: gcu_builder
   *     pt： gcu primitive type
   * Return：
   *     <max_op, min_op>
   * */
  static std::pair<builder::Op, builder::Op> GenerateNumericLimit(
      GcuBuilderPtr gcu_builder, GcuPrimitiveType pt);

  /*
   * Parameters:
   *     gcu_builder: gcu_builder
   *     type : gcu type
   * Return：
   *     const op
   * */
  static bool IsDyn(const std::vector<int64_t>& shape);

  static builder::Op GetConst(const GcuBuilderPtr& gcu_builder,
                              const GcuPrimitiveType& type,
                              const double& target);

  static void GraphToGcuExecutable(
      const std::string& program_key,
      const std::vector<ExecutablePtr>& exectuables,
      const std::vector<ResourceReflection>& reflections = {});

  static std::vector<ExecutablePtr> GetGcuExecutable(
      const std::string& program_key);

  static ResourceReflection GetExecutableRelections(
      const ExecutablePtr& exectuable);

  static void GcuRuntimeNodeToGraph(const std::vector<GraphPtr>& graphs,
                                    Node* node);

  static std::vector<GraphPtr> GetRuntimeNodeGraph(NodePtr node);

  static void GraphToGlobalMemoryRef(const std::string& program_key,
                                     const GlobalMemoryRef& ref);

  static GlobalMemoryRef GetGlobalMemoryRef(const std::string& program_key);

  static void TransAndRecordWeights(const std::string& var_name,
                                    const GcuTransInfo& info);
  static const std::map<std::string, GcuTransInfo>& GetTransedWeights();
  static const std::map<std::string, std::vector<std::string>>&
  GetOptimizerLinkageParam();
  static int64_t StringToNumber(const std::string& str);
};

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
