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

#include "paddle/fluid/pybind/pir_utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_attribute.h"

namespace paddle {
std::vector<std::string> GetFeedTargetNames(pir::Program *prog) {
  std::vector<std::string> feed_target;
  for (auto &op : *(prog->block())) {
    if (op.isa<paddle::dialect::DataOp>()) {
      auto name = op.attribute<pir::StrAttribute>("name").AsString();
      feed_target.push_back(name);
      continue;
    } else if (op.isa<paddle::dialect::FeedOp>()) {
      auto name = op.attribute<pir::StrAttribute>("name").AsString();
      feed_target.push_back(name);
      continue;
    }
  }
  return feed_target;
}

std::vector<std::string> GetFetchTargetNames(pir::Program *prog) {
  std::vector<std::string> fetch_target;
  for (auto &op : *(prog->block())) {
    if (op.isa<paddle::dialect::FetchOp>()) {
      auto name = op.attribute<pir::StrAttribute>("name").AsString();
      fetch_target.push_back(name);
      continue;
    } else if (op.isa<pir::ShadowOutputOp>()) {
      auto name = op.attribute<pir::StrAttribute>("output_name").AsString();
      fetch_target.push_back(name);
      continue;
    }
  }
  return fetch_target;
}
}  // namespace paddle
