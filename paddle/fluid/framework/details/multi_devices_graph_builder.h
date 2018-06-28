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

namespace paddle {
namespace platform {
class NCCLContextMap;
}

namespace framework {
class Scope;
namespace details {

class MultiDevSSAGraphBuilder : public SSAGraphBuilder {
 public:
#ifdef PADDLE_WITH_CUDA
  MultiDevSSAGraphBuilder(const std::vector<platform::Place> &places,
                          const std::string &loss_var_name,
                          const std::unordered_set<std::string> &params,
                          const std::vector<Scope *> &local_scopes,
                          platform::NCCLContextMap *nccl_ctxs,
                          const BuildStrategy &strategy);
#else
  MultiDevSSAGraphBuilder(const std::vector<platform::Place> &places,
                          const std::string &loss_var_name,
                          const std::unordered_set<std::string> &params,
                          const std::vector<Scope *> &local_scopes,
                          const BuildStrategy &strategy);
#endif

  std::unique_ptr<SSAGraph> Build(const ProgramDesc &program) const override;
  int GetVarDeviceID(const std::string &varname) const override;

 private:
  void CreateOpHandleIOs(SSAGraph *result, const OpDesc &op,
                         size_t device_id) const;

 private:
  std::string loss_var_name_;
  const std::vector<platform::Place> &places_;
  const std::vector<Scope *> &local_scopes_;
  std::unordered_set<std::string> grad_names_;

#ifdef PADDLE_WITH_CUDA
  platform::NCCLContextMap *nccl_ctxs_;
#endif

  bool IsScaleLossOp(const OpDesc &op) const;

  void CreateRPCOp(SSAGraph *result, const OpDesc &op) const;
  void CreateDistTrainOp(SSAGraph *result, const OpDesc &op) const;

  /**
   * Is this operator as the end-point operator before/after send operator.
   */
  bool IsDistTrainOp(const OpDesc &op,
                     const std::vector<std::string> &send_vars,
                     const std::vector<std::string> &recv_vars) const;

  std::vector<std::string> FindDistTrainSendVars(
      const ProgramDesc &program) const;

  std::vector<std::string> FindDistTrainRecvVars(
      const ProgramDesc &program) const;

  void ConnectOp(SSAGraph *result, OpHandleBase *op,
                 const std::string &prev_op_name) const;

  void CreateComputationalOps(SSAGraph *result, const OpDesc &op,
                              size_t num_places) const;

  void CreateScaleLossGradOp(SSAGraph *result) const;
  VarHandle *CreateReduceOp(SSAGraph *result, const std::string &og,
                            int dst_dev_id) const;
  void CreateComputationalOp(SSAGraph *result, const OpDesc &op,
                             int dev_id) const;

  bool IsParameterGradientOnce(
      const std::string &og,
      std::unordered_set<std::string> *og_has_been_broadcast) const;

  int GetOpDeviceID(const OpDesc &op) const;

  void InsertAllReduceOp(SSAGraph *result, const std::string &og) const;

  void CreateBroadcastOp(SSAGraph *result, const std::string &p_name,
                         size_t src_dev_id) const;

  bool IsSparseGradient(const std::string &og) const;

  size_t GetAppropriateDeviceID(
      const std::vector<std::string> &var_names) const;

 private:
  BuildStrategy strategy_;
  mutable std::unordered_map<std::string, VarDesc *> all_vars_;
  mutable std::unordered_map<std::string, int> var_name_on_devices_;
  mutable std::vector<int64_t> balance_vars_;

  void SetCommunicationContext(OpHandleBase *op_handle,
                               const platform::Place &p) const;
};
}  // namespace details
}  // namespace framework
}  // namespace paddle
