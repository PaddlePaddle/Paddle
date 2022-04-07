// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/custom_operator/custom_operator_node.h"
#include "paddle/fluid/framework/custom_operator.h"
#include "paddle/fluid/framework/op_meta_info_helper.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/core/dense_tensor.h"

namespace egr {
std::vector<std::vector<paddle::experimental::Tensor>> RunCustomOpNode::
operator()(std::vector<std::vector<paddle::experimental::Tensor>>& grads,
           bool create_graph) {  // NOLINT
  paddle::CustomOpKernelContext ctx;
  auto grad_inputs_name = paddle::framework::OpMetaInfoHelper::GetInputs(
      egr::Controller::Instance().GetOpMetaInfoMap().at(op_type_)[1]);
  auto grad_outputs_names = paddle::framework::OpMetaInfoHelper::GetOutputs(
      egr::Controller::Instance().GetOpMetaInfoMap().at(op_type_)[1]);
  auto map = egr::Controller::Instance().GetCustomEdgesSlotMap().at(op_type_);
  auto kernel_map = egr::Controller::Instance().GetOpMetaInfoMap();

  std::vector<std::vector<paddle::experimental::Tensor>> tmp_ins(
      grad_inputs_name.size());
  VLOG(7) << " Prepare Backward inputs of grads with size: " << grads.size()
          << ", whose grad_inputs_name size is: " << grad_inputs_name.size();
  for (size_t i = 0; i < grads.size(); i++) {
    if (map[1].find(i) != map[1].end()) {
      VLOG(7) << "Insert grad: " << i << " to grad_inputs: " << map[1][i];
      tmp_ins[map[1][i]] = grads[i];
    }
  }

  for (auto it : fwd_outs) {
    VLOG(7) << "Insert fwd_outs to grad_inputs: " << it.first;
    tmp_ins[it.first] = RunCustomOpNode::Recover(&(it.second));
  }

  for (auto it : fwd_ins) {
    VLOG(7) << "Insert fwd_ins to grad_inputs: " << it.first;
    tmp_ins[it.first] = RunCustomOpNode::Recover(&(it.second));
  }

  VLOG(6) << "Prepare Grad inputs";
  for (const auto& in : tmp_ins) {
    ctx.EmplaceBackInputs(in);
  }
  VLOG(6) << "Prepare Grad attrs";
  ctx.EmplaceBackAttrs(attrs_);
  std::vector<std::vector<paddle::experimental::Tensor>> outs(
      GetEdges().size());
  std::vector<std::vector<paddle::experimental::Tensor>> tmp_outs(
      grad_outputs_names.size());
  VLOG(6) << "Prepare Grad outputs for size: " << grad_outputs_names.size();
  for (size_t i = 0; i < GetEdges().size(); i++) {
    if (map[0].find(i) != map[0].end()) {
      VLOG(7) << "Insert grad outputs: " << i
              << " with size: " << GetEdges()[i].size()
              << " to tmp_outputs: " << map[0][i];
      for (size_t j = 0; j < GetEdges()[i].size(); j++) {
        outs[i].emplace_back(/* init it incase of copy nullptr of shared_ptr */
                             std::make_shared<phi::DenseTensor>(
                                 phi::DataType::UNDEFINED),
                             egr::Controller::Instance().GenerateUniqueName(
                                 "custom_tmp_grad"));
      }
      tmp_outs[map[0][i]] = outs[i];
    }
  }
  for (size_t i = 0; i < tmp_outs.size(); i++) {
    VLOG(7) << "Prepare grad outputs size: " << tmp_outs[i].size();
    ctx.EmplaceBackOutputs(tmp_outs[i]);
  }
  VLOG(7) << "Run Kernel of Grad Custom Op: " << op_type_;

  (*paddle::framework::OpMetaInfoHelper::GetKernelFn(
      kernel_map.at(op_type_)[1]))(&ctx);
  return outs;
}
}  // namespace egr
