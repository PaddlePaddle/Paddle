// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <popart/builder.hpp>
#include <popart/graphtransformer.hpp>
#include <popart/optimizer.hpp>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device/ipu/ipu_names.h"
#include "paddle/fluid/platform/device/ipu/ipu_strategy.h"
#include "paddle/fluid/platform/device/ipu/ipu_utils.h"

namespace paddle {
namespace platform {
namespace ipu {

struct CompilerResources {
  // popart input tensor_ids
  std::vector<popart::TensorId> inputs;
  // popart output tensor_ids
  std::vector<popart::TensorId> outputs;
  // <paddle_var_name, popart_tensor_ids>
  std::map<std::string, popart::TensorId> tensors;
  // popart_weight_ids
  std::vector<popart::TensorId> weights;
  // popart loss tensor_id
  popart::TensorId loss_var;
  // paddle lr var_name
  std::string lr_var;
  // lr value
  float lr;
  // flag for lr is constant or scheduling
  bool with_lr_sched = false;
  // paddle optimizer type, eg: momentum, lamb
  std::string optimizer_type;

  using OptimizerFn =
      std::function<std::unique_ptr<popart::Optimizer>(float lr)>;
  OptimizerFn optimizer_fn;
  // The eval mode of optimizer in training
  std::unique_ptr<popart::Optimizer> eval_optimizer;

 public:
  popart::Optimizer *Optimizer() { return optimizer.get(); }

  popart::Optimizer *NewOptimizer() {
    optimizer = optimizer_fn(lr);
    return optimizer.get();
  }

  popart::Optimizer *UpdateOptimizer(float lr_new) {
    optimizer = optimizer_fn(lr_new);
    return optimizer.get();
  }

 private:
  std::unique_ptr<popart::Optimizer> optimizer;
};

// helper for lowering graph
struct GraphHelper {
  explicit GraphHelper(const Graph *);

  const Graph *graph;
  std::map<std::string, Node *> vars_name_map;
  std::map<int, Node *> nodes_id_map;
  std::vector<Node *> sorted_ops;
  std::vector<int> sorted_vars_id;
};

class Compiler {
 public:
  Compiler();
  ~Compiler();

  void Prepare(const Graph *graph);
  void InitInputs(const std::vector<std::string> &feed_list);
  void InitOutputs(const std::vector<std::string> &fetch_list);
  void LowerConstants(const Scope *scope);
  void LowerWeights(const Scope *scope);
  void LowerBody();
  void LowerOptimizer(const Scope *scope);

  void SetIpuStrategy(const IpuStrategy &strategy) {
    ipu_strategy_ = &strategy;
  }

  void SetCustomOps(const std::vector<IpuCustomOpIdentifier> &custom_ops);

  CompilerResources *GetResources() { return resources_.get(); }

  std::string GetModelProto();
  std::string GetFP16ModelProto();

  void SaveModelProto(const std::string &path);
  void SaveModelProtoNoCheck(const std::string &path);

 private:
  void RegisterOpFunc();
  std::vector<std::string> GetOpInputs(const OpDesc *op);
  const std::vector<std::string> &GetOpOutputs(const OpDesc *op);
  const std::string GetNameScope(const OpDesc *op);
  popart::DebugContext BuildDebugContext(const OpDesc *op);

  void InsertTensors(const std::vector<std::string> &output_names,
                     const std::vector<std::string> &tensor_ids);
  void InsertTensors(const std::vector<std::string> &output_names,
                     const std::string &tensor_id);
  void SetIpuIndexStage(const std::vector<std::string> &tensor_ids,
                        const OpDesc *op_desc);
  void SetIpuIndexStage(const std::string &tensor_id, const OpDesc *op_desc);
  void SetAMPAttributes(const std::vector<std::string> &tensor_ids,
                        const OpDesc *op_desc);
  void SetAMPAttributes(const std::string &tensor_id, const OpDesc *op_desc);
  void SetSerializeAttributes(const std::vector<std::string> &tensor_ids,
                              const OpDesc *op_desc);
  void SetSerializeAttributes(const std::string &tensor_id,
                              const OpDesc *op_desc);
  void PushNameScope(const OpDesc *op);
  void PopNameScope(const OpDesc *op);

 private:
  std::unique_ptr<popart::Builder> builder_;
  std::unique_ptr<CompilerResources> resources_;
  std::unique_ptr<GraphHelper> graph_helper_;

  using OpFunc = std::function<void(OpDesc *op_desc)>;
  std::unordered_map<std::string, OpFunc> name_function_;

  const IpuStrategy *ipu_strategy_ = nullptr;
  std::map<std::string, IpuCustomOpIdentifier> custom_ops_;

  // Used to choose the way to set amp for Ops
  // If anyone op has the attr sAvailMemAttribute, the
  // available_memory_proportion from ipu_strategy
  // will be ignored and the Ops are set by their own sAvailMemAttribute. Else,
  // all relevant Ops will be set by
  // the available_memory_proportion from ipu_strategy.
  bool set_amp_for_all_ = true;
};

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
