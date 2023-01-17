/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {
class MemOptVarInfo;
class Node;
}  // namespace ir

namespace paddle2cinn {

constexpr char kCinnLaunchOp[] = "cinn_launch";
constexpr char kInputVars[] = "InputVars";
constexpr char kNoNeedBufferFeeds[] = "NoNeedBufferFeeds";
constexpr char kInternalVars[] = "InternalVars";
constexpr char kOutputVars[] = "OutputVars";
constexpr char kMemOptVarInfoFromMainGraph[] =
    "mem_opt_var_info_from_main_graph";
constexpr char kSkipGcVarNames[] = "skip_gc_vars";

using Name2VarInfoMap =
    std::unordered_map<std::string,
                       std::shared_ptr<framework::ir::MemOptVarInfo>>;
using GraphNodeSet = std::unordered_set<ir::Node*>;

// OpTransInfo contains informations used to detect subgraphs
// supported by the CINN compiler.
class OpTransInfo {
  using DyOpCondT =
      std::unordered_map<std::string, std::function<bool(const ir::Node&)>>;
  using DeParamCondT =
      std::unordered_map<std::string, std::unordered_set<std::string>>;

 public:
  OpTransInfo();

  const DyOpCondT& dynamic_op_cond() const { return dynamic_op_cond_; }

  const DeParamCondT& deny_param_cond() const { return deny_param_cond_; }

  const std::unordered_set<std::string>& default_deny_ops() const {
    return default_deny_ops_;
  }

  std::unordered_set<std::string> GetDenyVarNames(
      const GraphNodeSet& cluster) const;

  std::unordered_set<std::string> GetInplaceVarNames(
      const GraphNodeSet& cluster) const;

 private:
  DyOpCondT dynamic_op_cond_;

  DeParamCondT deny_param_cond_{{"batch_norm", {"ReserveSpace"}},
                                {"batch_norm_grad", {"ReserveSpace"}}};

  std::unordered_set<std::string> default_deny_ops_{"feed", "fetch"};
};

// A pass named BuildCinnPass, the function of this pass is:
//
// a) Detect the subgraphs that can be compiled by the CINN compiler. We call a
// detected subgraph a cluster, which is consisted of several op nodes.
//
// b) Call the CINN compiler to compile each original cluster and get the
// compiled cluster, which is consisted of several kCinnLaunchOp.
//
// c) Replace the original cluster with corresponding compiled cluster on the
// original graph.
//
// In this pass, some questions are handled with cautions:
//
// a) How to determine whether two op nodes can be divided into a cluster?
// Firstly, both op nodes should be compile supported.
// Secondly, there should be a direct path between the two op nodes through a
// var node.
// Thirdly, there should be no extra path between the two op nodes through
// unsupported op nodes.
// Lastly, if op nodes a and b can be divied into a cluster, op nodes b and c
// can be divided into a cluster, a and c can also be divided into a cluster.
// The implementation of cluster detection is encapsulated in the
// SubGraphDetector
// class.
//
// b) How to deal with the links between the var nodes in global graph and the
// op nodes in a cluster?
// We first add links between the var nodes in global graph and the op nodes in
// the compiled cluster, and then remove useless links between the var nodes in
// global graph and the op nodes in the original cluster.
class BuildCinnPass : public framework::ir::Pass {
 protected:
  void ApplyImpl(framework::ir::Graph* graph) const override;
};

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
