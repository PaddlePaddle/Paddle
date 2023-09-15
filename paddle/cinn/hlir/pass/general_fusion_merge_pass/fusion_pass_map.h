// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <memory>
#include <set>
#include <unordered_map>

#include "glog/logging.h"

#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/fuse_pass.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/input_fuse_pass.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/lightware_fuse_pass.h"

namespace cinn {
namespace hlir {
namespace pass {

struct LightwareFusePassComparator {
  bool operator()(const std::shared_ptr<LightwareFusePass>& lhs,
                  const std::shared_ptr<LightwareFusePass>& rhs) const {
    return lhs->Benefit() > rhs->Benefit();
  }
};

struct InputFusePassComparator {
  bool operator()(const std::shared_ptr<InputFusePass>& lhs,
                  const std::shared_ptr<InputFusePass>& rhs) const {
    return lhs->Benefit() > rhs->Benefit();
  }
};

class FusionPassMap {
 public:
  static FusionPassMap& Instance() {
    static FusionPassMap global_fusion_pass_map;
    return global_fusion_pass_map;
  }

  bool Has(const std::string& pass_name) const {
    return map_.find(pass_name) != map_.end();
  }

  void Insert(const std::string& pass_name,
              const std::shared_ptr<FusePass>& pass) {
    CHECK(!Has(pass_name)) << "FusePass " << pass_name
                           << " has already been registered.";
    map_.insert({pass_name, pass});
  }

  std::shared_ptr<FusePass> Get(const std::string& pass_name) const {
    auto it = map_.find(pass_name);
    CHECK(it != map_.end())
        << "FusePass " << pass_name << " has not been registered.";
    return it->second;
  }

  // fuse_mode: HorizontalFuse, VerticalFuse, RecomputeFuse
  std::vector<std::shared_ptr<LightwareFusePass>> GetLightwareFusePassesByMode(
      const std::string& fuse_mode) const {
    CHECK(fuse_mode == "HorizontalFuse" || fuse_mode == "VerticalFuse" ||
          fuse_mode == "RecomputeFuse")
        << "fuse_mode only supports HorizontalFuse, VerticalFuse and "
           "RecomputeFuse. Please check your input modes = "
        << fuse_mode;
    std::set<std::shared_ptr<LightwareFusePass>, LightwareFusePassComparator>
        candidate_passes;
    for (const auto iter : map_) {
      if (fuse_mode == iter.second->FuseMode()) {
        candidate_passes.insert(
            std::dynamic_pointer_cast<LightwareFusePass>(iter.second));
      }
    }
    return std::vector<std::shared_ptr<LightwareFusePass>>(
        candidate_passes.begin(), candidate_passes.end());
  }

  std::vector<std::shared_ptr<InputFusePass>> GetInputFusePasses() const {
    std::set<std::shared_ptr<InputFusePass>, InputFusePassComparator>
        candidate_passes;
    for (const auto iter : map_) {
      if (iter.second->FuseMode() == "InputFuse") {
        candidate_passes.insert(
            std::dynamic_pointer_cast<InputFusePass>(iter.second));
      }
    }
    return std::vector<std::shared_ptr<InputFusePass>>(candidate_passes.begin(),
                                                       candidate_passes.end());
  }

 private:
  FusionPassMap() = default;
  std::unordered_map<std::string, std::shared_ptr<FusePass>> map_;

  DISABLE_COPY_AND_ASSIGN(FusionPassMap);
};

}  // namespace pass
}  // namespace hlir
}  // namespace cinn
