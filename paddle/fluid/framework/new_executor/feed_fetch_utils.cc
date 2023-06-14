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

#include "paddle/fluid/framework/new_executor/feed_fetch_utils.h"

#include <map>
#include <vector>

#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {

void SetColAttrForFetchOps(const interpreter::Job& job,
                           std::shared_ptr<ProgramDesc> program_desc) {
  const std::set<std::string>& valid_feed_fetch_op_types = {"fetch",
                                                            "fetch_v2"};

  const std::vector<int> all_op_ids = job.AllFetchOpIds();
  for (int op_id : all_op_ids) {
    int col_attr = job.ColAttrForFetchOp(op_id);
    OpDesc* op_desc = program_desc->MutableBlock(0)->Op(op_id);
    PADDLE_ENFORCE(valid_feed_fetch_op_types.find(op_desc->Type()) !=
                       valid_feed_fetch_op_types.end(),
                   phi::errors::InvalidArgument(
                       "Op (%s) corressponding to feed_fetch_op_id (%d) is not "
                       "in valid_feed_fetch_op_types.",
                       op_desc->Type(),
                       op_id));

    op_desc->SetAttr("col", col_attr);
  }
}

}  // namespace framework
}  // namespace paddle
