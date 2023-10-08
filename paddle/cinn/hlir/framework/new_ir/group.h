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

#include "paddle/cinn/hlir/framework/new_ir/utils.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/pir/core/operation.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace newir {
using framework::OpPatternKind;

// TODO(Aurelius84): Need to be replaced with CinnGroupOp
struct Group {
 public:
  explicit Group(const std::vector<::pir::Operation*>& group_ops)
      : ops(group_ops) {
    Initialize();
  }

  explicit Group(std::initializer_list<::pir::Operation*> group_ops)
      : ops(group_ops) {
    Initialize();
  }

  int group_id;
  std::string fn_name;
  OpPatternKind op_pattern_kind;
  std::vector<::pir::Operation*> ops;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

 private:
  void Initialize() {
    op_pattern_kind = OpPatternKind::kElementWise;
    fn_name = CompatibleInfo::GroupOpsName(ops);
  }
};

}  // namespace newir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
