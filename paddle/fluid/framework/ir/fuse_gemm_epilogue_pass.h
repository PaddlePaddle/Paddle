// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.
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

#include <mutex>
#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * Fuse the ElewiseAdd and activation
 */
class Graph;
class Node;

class EpiloguePassActivationCache {
 public:
  static EpiloguePassActivationCache &Instance() {
    static EpiloguePassActivationCache instance;
    return instance;
  }

  EpiloguePassActivationCache(const EpiloguePassActivationCache &) = delete;
  void operator=(const EpiloguePassActivationCache &) = delete;

  bool HasFusedActivation(const std::string &key) {
    return fused_activation_keys.count(key);
  }

  void InsertFusedActivation(const std::string &key) {
    if (!HasFusedActivation(key)) {
      mtx.lock();
      fused_activation_keys.insert(key);
      mtx.unlock();
    }
  }

 private:
  EpiloguePassActivationCache() {}
  std::unordered_set<std::string> fused_activation_keys;
  std::mutex mtx;
};

class FuseGemmEpiloguePass : public FusePassBase {
 public:
  virtual ~FuseGemmEpiloguePass() {}

 protected:
  void ApplyImpl(ir::Graph *graph) const override;

  ir::Graph *FuseLinearFwd(ir::Graph *graph, bool is_training) const;
  ir::Graph *FuseLinearActFwd(ir::Graph *graph,
                              const std::unordered_set<std::string> &act_types,
                              bool is_training) const;
  ir::Graph *FuseLinearBwd(ir::Graph *graph, bool is_first_gemm) const;
  ir::Graph *FuseLinearActBwd(
      ir::Graph *graph,
      const std::unordered_set<std::string> &act_grad_types) const;

 private:
  bool IsGemmFromLinear_(const std::vector<int64_t> &x_shape,
                         const std::vector<int64_t> &w_shape,
                         OpDesc *matmul_v2_op) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
