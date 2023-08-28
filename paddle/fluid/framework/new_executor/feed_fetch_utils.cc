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

void SetColAttrForFeedFetchOps(std::shared_ptr<ProgramDesc> program_desc,
                               const int64_t micro_batch_num,
                               const int64_t micro_batch_id) {
  const std::set<std::string>& valid_feed_fetch_op_types = {
      "fetch", "fetch_v2", "feed"};
  for (const auto& op_desc : program_desc->MutableBlock(0)->AllOps()) {
    if (valid_feed_fetch_op_types.find(op_desc->Type()) !=
        valid_feed_fetch_op_types.end()) {
      int col = op_desc->GetAttrIfExists<int>("col");
      PADDLE_ENFORCE_GE(
          col,
          0,
          platform::errors::InvalidArgument(
              "Expected the column index (the attribute 'col' of "
              "operator 'Fetch') of current fetching variable to be "
              "no less than 0. But received column index = %d.",
              col));
      int new_col = static_cast<int>(col * micro_batch_num + micro_batch_id);
      op_desc->SetAttr("col", new_col);
      VLOG(6) << "Job (" << micro_batch_id << ") Set " << op_desc->Type()
              << "'s attr col=" << new_col;
    }
  }
}

}  // namespace framework
}  // namespace paddle
