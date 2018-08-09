//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/ssa_graph_builder.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace platform {
class NCCLContextMap;
}

namespace framework {
class Scope;
namespace details {

class MultiDevSSAGraphBuilder : public SSAGraphBuilder {
 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override;

 private:
  void CreateOpHandleIOs(ir::Graph *result, ir::Node *node,
                         size_t device_id) const;
  void Init() const;

 private:
  mutable std::string loss_var_name_;
  mutable std::vector<platform::Place> places_;
  mutable std::vector<Scope *> local_scopes_;
  mutable std::unordered_set<std::string> grad_names_;

#ifdef PADDLE_WITH_CUDA
  mutable platform::NCCLContextMap *nccl_ctxs_;
#endif

  int GetVarDeviceID(const ir::Graph &graph, const std::string &varname) const;

  bool IsScaleLossOp(ir::Node *node) const;

  void CreateRPCOp(ir::Graph *result, ir::Node *node) const;
  void CreateDistTrainOp(ir::Graph *result, ir::Node *node) const;

  /**
   * Is this operator as the end-point operator before/after send operator.
   */
  bool IsDistTrainOp(ir::Node *node, const std::vector<std::string> &send_vars,
                     const std::vector<std::string> &recv_vars) const;

  std::vector<std::string> FindDistTrainSendVars(
      const std::vector<ir::Node *> &nodes) const;

  std::vector<std::string> FindDistTrainRecvVars(
      const std::vector<ir::Node *> &nodes) const;

  void ConnectOp(ir::Graph *result, OpHandleBase *op,
                 const std::string &prev_op_name) const;

  void CreateComputationalOps(ir::Graph *result, ir::Node *node,
                              size_t num_places) const;

  void CreateScaleLossGradOp(ir::Graph *result,
                             const std::string &loss_grad_name) const;

  VarHandle *CreateReduceOp(ir::Graph *result, const std::string &og,
                            int dst_dev_id) const;
  void CreateComputationalOp(ir::Graph *result, ir::Node *node,
                             int dev_id) const;

  bool IsParameterGradientOnce(
      const std::string &og,
      std::unordered_set<std::string> *og_has_been_broadcast) const;

  int GetOpDeviceID(const ir::Graph &graph, ir::Node *node) const;

  void InsertAllReduceOp(ir::Graph *result, const std::string &og) const;

  void InsertDataBalanceOp(ir::Graph *result,
                           const std::vector<std::string> &datas) const;

  void CreateBroadcastOp(ir::Graph *result, const std::string &p_name,
                         size_t src_dev_id) const;

  bool IsSparseGradient(const std::string &og) const;

  size_t GetAppropriateDeviceID(
      const std::vector<std::string> &var_names) const;

 private:
  mutable BuildStrategy strategy_;
  mutable std::unordered_map<std::string, VarDesc *> all_vars_;
  mutable std::vector<int64_t> balance_vars_;

  void SetCommunicationContext(OpHandleBase *op_handle,
                               const platform::Place &p) const;
};
}  // namespace details
}  // namespace framework
}  // namespace paddle
