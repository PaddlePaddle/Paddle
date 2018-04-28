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
#include <vector>

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
                          bool skip_scale_loss,
                          platform::NCCLContextMap *nccl_ctxs);
#else
  MultiDevSSAGraphBuilder(const std::vector<platform::Place> &places,
                          const std::string &loss_var_name,
                          const std::unordered_set<std::string> &params,
                          const std::vector<Scope *> &local_scopes,
                          bool skip_scale_loss);
#endif

  std::unique_ptr<SSAGraph> Build(const ProgramDesc &program) const override;

 private:
  void CreateOpHandleIOs(SSAGraph *result, const OpDesc &op,
                         size_t place_id) const;

 private:
  std::string loss_var_name_;
  const std::vector<platform::Place> &places_;
  const std::vector<Scope *> &local_scopes_;
  std::unordered_set<std::string> grad_names_;

#ifdef PADDLE_WITH_CUDA
  platform::NCCLContextMap *nccl_ctxs_;
#endif
  bool skip_scale_loss_;

  bool IsScaleLossOp(const OpDesc &op) const;

  void CreateSendOp(SSAGraph *result, const OpDesc &op) const;

  /**
   * Is this operator as the end-point operator before/after send operator.
   */
  bool IsDistTrainOp(const OpDesc &op, OpDesc *send_op) const;

  void CreateComputationalOps(SSAGraph *result, const OpDesc &op,
                              size_t num_places) const;

  void CreateScaleLossGradOp(SSAGraph *result) const;

  bool IsParameterGradientOnce(
      const std::string &og,
      std::unordered_set<std::string> *og_has_been_broadcast) const;

  void InsertNCCLAllReduceOp(SSAGraph *result, const std::string &og) const;

  /**
   * Get send op in the global block of program.
   * nullptr if not found.
   */
  OpDesc *GetSendOpDesc(const ProgramDesc &program) const;
};
}  // namespace details
}  // namespace framework
}  // namespace paddle
