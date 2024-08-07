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
#include <unordered_map>
#include <unordered_set>
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/utils/type_defs.h"
#include "paddle/common/ddim.h"
#include "paddle/pir/include/core/operation.h"

namespace cinn {
namespace hlir {
namespace framework {

namespace pir {

struct CINNKernelInfo {
  std::string fn_name;
  void* fn_ptr;
  void* infer_shape_fn_ptr;
  void* CX86_fn_ptr;

  struct ArgDimIdx {
    int arg_idx;
    int dim_idx;
  };
  // int_args_map records the int_args_map.key argument (dtype is Int) in the
  // kernel parameter taken from the dim_idx dimension of the shape of the
  // ArgDimIdx.arg_idx argument.
  // Examples:
  //   a func like: foo(tensor A, tensor B, int S1, int S2)
  //   S1 = A.shape[3]
  //   S2 = B.shape[2]
  //   int_args_map will be like
  //   {
  //     2: {0, 3},
  //     3: {1, 2}
  //   }
  std::map<int, ArgDimIdx> int_args_map;
};

struct CompatibleInfo {
  static constexpr char* kNamePrefix = "var";
  // NOTE(Aurelius): Some ops in CINN register different
  // name between OpMapper and Compute/Schedule, such as
  // 'subtract': 1. OpMapper: 'elementwise_sub'; 2. Compute/Schedule:
  // 'subtract'.
  static const std::unordered_map<std::string, std::string> OP_NAMES;

  static const std::unordered_set<std::string> TOCINN_OPS;

  static bool IsDeniedForCinn(const ::pir::Operation& op);

  static bool IsSupportForCinn(const ::pir::Operation& op);

  static std::string OpName(const ::pir::Operation& op);

  static std::string OpFuncName(const ::pir::Operation& op);

  static std::string GroupOpsName(const std::vector<::pir::Operation*>& ops);

  static std::string ValueName(const ::pir::Value& value);

  static std::vector<::pir::Value> RealOperandSources(
      const ::pir::Operation& op);

  static utils::Attribute ConvertAttribute(const ::pir::Attribute& src_attr);

  static utils::AttributeMap ConvertAttributes(const ::pir::Operation& op);

  static cinn::common::Type ConvertIRType(::pir::Type type);

  static std::vector<int> ValueShape(const ::pir::Value& value);

  static int ShapeProduct(const std::vector<int>& shape);

  static OpPatternKind OpKind(const ::pir::Operation& op);
};

std::vector<int64_t> GetBroadcastAxis(const ::common::DDim& in_shape,
                                      const std::vector<int64_t>& out_shape);

class PrettyNamer {
 public:
  const std::string& GetOrNew(::pir::Value hash_key,
                              const std::string& name_hint) {
    if (pretty_names_.find(hash_key) == pretty_names_.end()) {
      pretty_names_[hash_key] = name_generator_.New(name_hint);
    }
    return pretty_names_.at(hash_key);
  }

  ::cinn::common::NameGenerator& GetNameGenerator() { return name_generator_; }

  const std::unordered_map<::pir::Value, std::string>& GetNameMap() {
    return pretty_names_;
  }

 private:
  std::unordered_map<::pir::Value, std::string> pretty_names_;
  ::cinn::common::NameGenerator name_generator_;
};

enum class ScheduleAlignType : int {
  kNone = 0,       //! No need to align
  kBroadcast = 1,  //! Using Broadcast schedule to align
};

struct ScheduleInfoNode {
  // TOOD(phlrain): update align type by new loop alignment
  ScheduleAlignType type{ScheduleAlignType::kNone};

  // reduction or broadcast axis locations
  std::vector<int64_t> axis_info;
  // representing the iteration space
  std::vector<int64_t> factor_info;

  std::string DebugStr() const {
    std::stringstream ss;

    ss << "type  " << static_cast<int>(type) << "| axis info ";
    for (auto d : axis_info) {
      ss << " " << d;
    }
    ss << " | factor info ";
    for (auto f : factor_info) {
      ss << " " << f;
    }
    ss << "\n";

    return ss.str();
  }
};

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
