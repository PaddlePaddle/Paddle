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
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/core/dense_tensor.h"

namespace egr {

static void ConstructFwdAndBwdMap(
    const std::vector<paddle::OpMetaInfo>& vec_map,
    const std::string& op_type) {
  auto& in_out_map = egr::Controller::Instance().GetCustomEdgesSlotMap();
  if (in_out_map.find(op_type) != in_out_map.end()) {
    if (in_out_map[op_type].size() == 2) {
      VLOG(7) << "Find Exist CustomEdgesSlotMap Skip >>>> ";
      return;
    }
  }

  VLOG(7) << "Construct DoubleGrad's CustomEdgesSlotMap ";
  auto inputs_names =
      paddle::framework::OpMetaInfoHelper::GetInputs(vec_map[1]);
  auto outputs_names =
      paddle::framework::OpMetaInfoHelper::GetOutputs(vec_map[1]);
  auto attrs_names = paddle::framework::OpMetaInfoHelper::GetAttrs(vec_map[1]);
  auto grad_outputs_names =
      paddle::framework::OpMetaInfoHelper::GetOutputs(vec_map[2]);
  auto grad_inputs_names =
      paddle::framework::OpMetaInfoHelper::GetInputs(vec_map[2]);
  auto grad_attrs_names =
      paddle::framework::OpMetaInfoHelper::GetAttrs(vec_map[2]);
  std::vector<std::unordered_map<int, int>> res(5);
  in_out_map[op_type].push_back(res);
  // Prepare pos map for grad_outputs
  VLOG(7) << "Prepare pos map for grad_outputs";
  PADDLE_ENFORCE_LE(
      grad_outputs_names.size(),
      inputs_names.size(),
      paddle::platform::errors::InvalidArgument(
          "Grad outputs num should be less equal than forward inputs num."));
  for (size_t i = 0; i < grad_outputs_names.size(); i++) {
    auto end = grad_outputs_names[i].find("@GRAD@GRAD");
    if (end != std::string::npos) {
      for (size_t j = 0; j < inputs_names.size(); j++) {
        if (grad_outputs_names[i].substr(0, end + 5) == inputs_names[j]) {
          VLOG(7) << " ==== Custom Operator: " << op_type << "_grad "
                  << "'s No." << j << " inputs: " << inputs_names[j]
                  << " related to No." << i
                  << " grad_outputs: " << grad_outputs_names[i];
          in_out_map[op_type][1][0][j] = i;
        }
      }
    } else {
      size_t end_n = grad_outputs_names[i].find("@GRAD@NEW");
      if (end_n != std::string::npos) {
        for (size_t j = 0; j < inputs_names.size(); j++) {
          if (grad_outputs_names[i].substr(0, end_n) == inputs_names[j]) {
            VLOG(7) << " ==== Custom Operator: " << op_type << "_grad "
                    << "'s No." << j << " inputs: " << inputs_names[j]
                    << " related to No." << i
                    << " grad_outputs: " << grad_outputs_names[i];
            in_out_map[op_type][1][0][j] = i;
          }
        }
      } else {
        size_t end_one_grad = grad_outputs_names[i].find("@GRAD");
        if (end_one_grad != std::string::npos) {
          for (size_t j = 0; j < inputs_names.size(); j++) {
            if (grad_outputs_names[i].substr(0, end_one_grad) ==
                inputs_names[j]) {
              VLOG(7) << " ==== Custom Operator: " << op_type << "_grad "
                      << "'s No." << j << " inputs: " << inputs_names[j]
                      << " related to No." << i
                      << " grad_outputs: " << grad_outputs_names[i];
              in_out_map[op_type][1][0][j] = i;
            }
          }
        } else {
          PADDLE_THROW(paddle::platform::errors::NotFound(
              "All Grad outputs should be end of @GRAD@GRAD or @GRAD@NEW or "
              "@GRAD and we got %s is not one of them, "
              "please check your op and change to fit the rule.",
              grad_outputs_names[i]));
        }
      }
    }
  }
  // Prepare pos map for grad_inputs
  for (size_t i = 0; i < grad_inputs_names.size(); i++) {
    size_t end = grad_inputs_names[i].find("@GRAD@GRAD");
    if (end != std::string::npos) {
      for (size_t j = 0; j < outputs_names.size(); j++) {
        if (grad_inputs_names[i].substr(0, end + 5) == outputs_names[j]) {
          VLOG(7) << " ==== Custom Operator: " << op_type << "_grad "
                  << "'s No." << j << " outputs: " << outputs_names[j]
                  << " related to No." << i
                  << " grad_inputs's grad: " << grad_inputs_names[i];
          in_out_map[op_type][1][1][j] = i;
        }
      }
    } else {
      if (std::find(outputs_names.begin(),
                    outputs_names.end(),
                    grad_inputs_names[i]) != outputs_names.end()) {
        for (size_t j = 0; j < outputs_names.size(); j++) {
          if (grad_inputs_names[i] == outputs_names[j]) {
            VLOG(7) << " ==== Custom Operator: " << op_type << "_grad "
                    << "'s No." << j << " outputs: " << outputs_names[j]
                    << " related to No." << i
                    << " grad_inputs fwd outputs: " << grad_inputs_names[i];
            in_out_map[op_type][1][2][j] = i;
          }
        }
      } else {
        for (size_t j = 0; j < inputs_names.size(); j++) {
          if (grad_inputs_names[i] == inputs_names[j]) {
            VLOG(7) << " ==== Custom Operator: " << op_type << "_grad "
                    << "'s No." << j << " inputs: " << inputs_names[j]
                    << " related to No." << i
                    << " grad_inputs fwd inputs: " << grad_inputs_names[i];
            in_out_map[op_type][1][3][j] = i;
          }
        }
      }
    }
  }

  // Prepare pos map for grad attrs_
  for (size_t i = 0; i < grad_attrs_names.size(); i++) {
    auto end =
        std::find(attrs_names.begin(), attrs_names.end(), grad_attrs_names[i]);
    PADDLE_ENFORCE_NE(end,
                      attrs_names.end(),
                      paddle::platform::errors::NotFound(
                          "All Grad attrs should be one of forward attrs and "
                          "we got %s is not one of them, please check your "
                          "op and change to fit the rule.",
                          grad_attrs_names[i]));
    for (size_t j = 0; j < attrs_names.size(); j++) {
      if (grad_attrs_names[i] == attrs_names[j]) {
        VLOG(7) << " ==== Custom Operator: " << op_type << "_grad "
                << "'s No." << j << " attrs: " << attrs_names[j]
                << " related to No." << i
                << " grad_attrs: " << grad_attrs_names[i];
        in_out_map[op_type][1][4][j] = i;
      }
    }
  }
}

paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                     kSlotSmallVectorSize>
RunCustomOpNode::operator()(
    paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                         kSlotSmallVectorSize>& grads,
    bool create_graph,
    bool is_new_grad) {  // NOLINT
  paddle::CustomOpKernelContext ctx;
  auto grad_inputs_name = paddle::framework::OpMetaInfoHelper::GetInputs(
      egr::Controller::Instance().GetOpMetaInfoMap().at(op_type_)[1]);
  auto grad_outputs_names = paddle::framework::OpMetaInfoHelper::GetOutputs(
      egr::Controller::Instance().GetOpMetaInfoMap().at(op_type_)[1]);
  auto map = egr::Controller::Instance().GetCustomEdgesSlotMap().at(op_type_);
  auto kernel_map = egr::Controller::Instance().GetOpMetaInfoMap();

  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       kSlotSmallVectorSize>
      tmp_ins(grad_inputs_name.size());
  VLOG(7) << " Prepare Backward inputs of grads with size: " << grads.size()
          << ", whose grad_inputs_name size is: " << grad_inputs_name.size();
  auto hooked_grads = ApplyGradientHooks(grads);
  for (size_t i = 0; i < hooked_grads.size(); i++) {
    if (map[0][1].find(i) != map[0][1].end()) {
      VLOG(7) << "Insert grad: " << i << " to grad_inputs: " << map[0][1][i];
      tmp_ins[map[0][1][i]] = hooked_grads[i];
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
  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       kSlotSmallVectorSize>
      outs(OutputMeta().size());
  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       kSlotSmallVectorSize>
      tmp_outs(grad_outputs_names.size());
  VLOG(6) << "Prepare Grad outputs for size: " << grad_outputs_names.size();
  for (size_t i = 0; i < OutputMeta().size(); i++) {
    if (map[0][0].find(i) != map[0][0].end()) {
      int grad_output_idx = map[0][0][i];
      VLOG(7) << "Insert grad outputs: " << i
              << " with size: " << OutputMeta()[grad_output_idx].size()
              << " to tmp_outputs: " << grad_output_idx;
      for (size_t j = 0; j < OutputMeta()[grad_output_idx].size(); j++) {
        outs[grad_output_idx]
            .emplace_back(/* init it incase of copy nullptr of shared_ptr */
                          std::make_shared<phi::DenseTensor>(
                              phi::DataType::UNDEFINED),
                          egr::Controller::Instance().GenerateUniqueName(
                              "custom_tmp_grad"));
        egr::EagerUtils::autograd_meta(&(outs[grad_output_idx][j]));
      }
      tmp_outs[grad_output_idx] = outs[grad_output_idx];
    }
  }
  for (size_t i = 0; i < tmp_outs.size(); i++) {
    VLOG(7) << "Prepare grad outputs size: " << tmp_outs[i].size();
    ctx.EmplaceBackOutputs(tmp_outs[i]);
  }
  VLOG(7) << "Run Kernel of Grad Custom Op: " << op_type_ << "_grad";

  (*paddle::framework::OpMetaInfoHelper::GetKernelFn(
      kernel_map.at(op_type_)[1]))(&ctx);

  VLOG(7) << "Get AutogradMeta for inputs and outputs for Custom Op";
  std::vector<std::vector<egr::AutogradMeta*>> ins_auto_grad_metas;
  std::vector<std::vector<egr::AutogradMeta*>> outs_auto_grad_metas;
  VLOG(7) << "We got slot num of ins is: " << ctx.InputRange().size();
  ins_auto_grad_metas.resize(ctx.InputRange().size());
  VLOG(7) << "We got slot num of outs is: " << ctx.OutputRange().size();
  outs_auto_grad_metas.resize(ctx.OutputRange().size());

  for (size_t i = 0; i < ctx.InputRange().size(); i++) {
    ins_auto_grad_metas[i] =
        egr::EagerUtils::nullable_autograd_meta(ctx.InputsBetween(
            ctx.InputRangeAt(i).first, ctx.InputRangeAt(i).second));
  }

  for (size_t i = 0; i < ctx.OutputRange().size(); i++) {
    outs_auto_grad_metas[i] =
        egr::EagerUtils::unsafe_autograd_meta(ctx.OutputsBetweeen(
            ctx.OutputRangeAt(i).first, ctx.OutputRangeAt(i).second));
  }
  bool require_any_grad = false;
  bool trace_backward = egr::Controller::Instance().HasGrad() && create_graph;
  for (size_t i = 0; i < ins_auto_grad_metas.size(); i++) {
    require_any_grad =
        require_any_grad || egr::EagerUtils::ComputeRequireGrad(
                                trace_backward, &(ins_auto_grad_metas[i]));
  }

  auto meta_info_map = egr::Controller::Instance().GetOpMetaInfoMap();
  const auto& vec_map = meta_info_map.at(op_type_);
  if (require_any_grad && (vec_map.size() > 2)) {
    paddle::platform::RecordEvent node_creation_record_event(
        "Custom Op " + op_type_ + " double_grad node_creation",
        paddle::platform::TracerEventType::OperatorInner,
        1);
    VLOG(6) << " Construct Grad for Custom Op: " << op_type_;
    ConstructFwdAndBwdMap(vec_map, op_type_);
    for (size_t i = 0; i < outs_auto_grad_metas.size(); i++) {
      egr::EagerUtils::PassStopGradient(false, &(outs_auto_grad_metas[i]));
    }
    auto grad_node = std::make_shared<egr::RunCustomOpDoubleGradNode>(
        outs_auto_grad_metas.size(), ins_auto_grad_metas.size(), op_type_);

    auto slot_map =
        egr::Controller::Instance().GetCustomEdgesSlotMap().at(op_type_);
    // Prepare Grad outputs
    size_t no_grad_cnt = 0;
    for (size_t i = 0; i < ins_auto_grad_metas.size(); i++) {
      const std::vector<paddle::experimental::Tensor>& in_tensors =
          ctx.InputsBetween(ctx.InputRangeAt(i).first,
                            ctx.InputRangeAt(i).second);

      if (slot_map[1][0].find(i) != slot_map[1][0].end()) {
        grad_node->SetGradOutMeta(in_tensors, slot_map[1][0][i]);
      } else {
        grad_node->SetGradOutMeta(in_tensors,
                                  ins_auto_grad_metas.size() - 1 - no_grad_cnt);
        no_grad_cnt++;
      }
    }

    // Prepare Grad inputs with grad of fwd outputs
    for (size_t i = 0; i < outs_auto_grad_metas.size(); i++) {
      const std::vector<paddle::experimental::Tensor>& out_tensors =
          ctx.OutputsBetweeen(ctx.OutputRangeAt(i).first,
                              ctx.OutputRangeAt(i).second);
      egr::EagerUtils::SetOutRankWithSlot(&(outs_auto_grad_metas[i]), i);
      egr::EagerUtils::SetHistory(&(outs_auto_grad_metas[i]), grad_node);
      grad_node->SetGradInMeta(out_tensors, i);
      egr::EagerUtils::CheckAndRetainGrad(out_tensors);
    }

    // Prepare Grad inputs with fwd outputs
    for (auto it = slot_map[1][2].begin(); it != slot_map[1][2].end(); it++) {
      VLOG(7) << "Prepare fwd_outs: " << it->first
              << " to grad_inputs: " << it->second;
      grad_node->fwd_outs[it->second] =
          egr::RunCustomOpNode::ConstructTensorWrapper(
              ctx.OutputsBetweeen(ctx.OutputRangeAt(it->first).first,
                                  ctx.OutputRangeAt(it->first).second));
    }

    // Prepare Grad inputs with fwd inputs
    for (auto it = slot_map[1][3].begin(); it != slot_map[1][3].end(); it++) {
      VLOG(7) << "Prepare fwd_ins: " << it->first
              << " to grad_inputs: " << it->second;
      grad_node->fwd_ins[it->second] =
          egr::RunCustomOpNode::ConstructTensorWrapper(
              ctx.InputsBetween(ctx.InputRangeAt(it->first).first,
                                ctx.InputRangeAt(it->first).second));
    }

    auto attrs_names = paddle::framework::OpMetaInfoHelper::GetAttrs(
        meta_info_map.at(op_type_)[2]);
    std::vector<paddle::any> attrs(attrs_names.size());
    // Prepare attrs for Grad node
    for (auto it = slot_map[1][4].begin(); it != slot_map[1][4].end(); it++) {
      VLOG(7) << "Prepare fwd attrs: " << it->first
              << " to grad_attrs: " << it->second;
      attrs[it->second] = attrs_[it->first];
    }
    grad_node->SetAttrs(attrs);
  }

  return outs;
}

paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                     kSlotSmallVectorSize>
RunCustomOpDoubleGradNode::operator()(
    paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                         kSlotSmallVectorSize>& grads,
    bool create_graph,
    bool is_new_grad) {  // NOLINT
  paddle::CustomOpKernelContext ctx;
  auto meta_info_map = egr::Controller::Instance().GetOpMetaInfoMap();
  const auto& vec_map = meta_info_map.at(op_type_);
  auto grad_inputs_name =
      paddle::framework::OpMetaInfoHelper::GetInputs(vec_map[2]);
  auto grad_outputs_names =
      paddle::framework::OpMetaInfoHelper::GetOutputs(vec_map[2]);
  auto map = egr::Controller::Instance().GetCustomEdgesSlotMap().at(op_type_);
  auto kernel_map = egr::Controller::Instance().GetOpMetaInfoMap();

  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       kSlotSmallVectorSize>
      tmp_ins(grad_inputs_name.size());
  VLOG(7) << " Prepare Backward inputs of grads with size: " << grads.size()
          << ", whose grad_inputs_name size is: " << grad_inputs_name.size();

  auto hooked_grads = ApplyGradientHooks(grads);

  for (size_t i = 0; i < hooked_grads.size(); i++) {
    if (map[1][1].find(i) != map[1][1].end()) {
      VLOG(7) << "Insert grad: " << i << " to grad_inputs: " << map[1][1][i];
      tmp_ins[map[1][1][i]] = hooked_grads[i];
    }
  }

  for (auto it : fwd_outs) {
    VLOG(7) << "Insert fwd_outs to grad_inputs: " << it.first;
    tmp_ins[it.first] = RunCustomOpDoubleGradNode::Recover(&(it.second));
  }

  for (auto it : fwd_ins) {
    VLOG(7) << "Insert fwd_ins to grad_inputs: " << it.first;
    tmp_ins[it.first] = RunCustomOpDoubleGradNode::Recover(&(it.second));
  }

  VLOG(6) << "Prepare Grad inputs";
  for (const auto& in : tmp_ins) {
    ctx.EmplaceBackInputs(in);
  }
  VLOG(6) << "Prepare Grad attrs";
  ctx.EmplaceBackAttrs(attrs_);
  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       kSlotSmallVectorSize>
      outs(OutputMeta().size());
  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       kSlotSmallVectorSize>
      tmp_outs(grad_outputs_names.size());
  VLOG(6) << "Prepare Grad outputs for size: " << grad_outputs_names.size();

  for (const auto& name : grad_outputs_names) {
    VLOG(6) << "Prepare Grad outputs name is: " << name;
  }

  for (size_t i = 0; i < OutputMeta().size(); i++) {
    if (map[1][0].find(i) != map[1][0].end()) {
      VLOG(7) << "Insert grad outputs: " << i
              << " with size: " << OutputMeta()[i].size()
              << " to tmp_outputs: " << map[1][0][i];
      for (size_t j = 0; j < OutputMeta()[i].size(); j++) {
        outs[i].emplace_back(/* init it incase of copy nullptr of shared_ptr */
                             std::make_shared<phi::DenseTensor>(
                                 phi::DataType::UNDEFINED),
                             egr::Controller::Instance().GenerateUniqueName(
                                 "custom_tmp_grad"));
      }
      tmp_outs[map[1][0][i]] = outs[i];
    }
  }
  for (size_t i = 0; i < tmp_outs.size(); i++) {
    VLOG(7) << "Prepare grad outputs size: " << tmp_outs[i].size();
    ctx.EmplaceBackOutputs(tmp_outs[i]);
  }
  VLOG(7) << "Run Kernel of Grad Custom Op: " << name();

  (*paddle::framework::OpMetaInfoHelper::GetKernelFn(
      kernel_map.at(op_type_)[2]))(&ctx);

  return outs;
}
}  // namespace egr
