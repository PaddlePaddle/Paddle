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

#include "paddle/fluid/eager/custom_operator/custom_operator_utils.h"
#include "paddle/fluid/framework/custom_operator.h"
#include "paddle/fluid/framework/custom_operator_utils.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

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
  auto inputs_names = paddle::OpMetaInfoHelper::GetInputs(vec_map[1]);
  auto outputs_names = paddle::OpMetaInfoHelper::GetOutputs(vec_map[1]);
  auto attrs_names = paddle::OpMetaInfoHelper::GetAttrs(vec_map[1]);
  auto grad_outputs_names = paddle::OpMetaInfoHelper::GetOutputs(vec_map[2]);
  auto grad_inputs_names = paddle::OpMetaInfoHelper::GetInputs(vec_map[2]);
  auto grad_attrs_names = paddle::OpMetaInfoHelper::GetAttrs(vec_map[2]);
  std::vector<std::unordered_map<int, int>> res(5);
  in_out_map[op_type].push_back(res);
  // Prepare pos map for grad_outputs
  VLOG(7) << "Prepare pos map for grad_outputs";
  PADDLE_ENFORCE_LE(
      grad_outputs_names.size(),
      inputs_names.size(),
      common::errors::InvalidArgument(
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
          in_out_map[op_type][1][0][j] = i;  // NOLINT
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
            in_out_map[op_type][1][0][j] = i;  // NOLINT
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
              in_out_map[op_type][1][0][j] = i;  // NOLINT
            }
          }
        } else {
          PADDLE_THROW(common::errors::NotFound(
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
          in_out_map[op_type][1][1][j] = i;  // NOLINT
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
            in_out_map[op_type][1][2][j] = i;  // NOLINT
          }
        }
      } else {
        for (size_t j = 0; j < inputs_names.size(); j++) {
          if (grad_inputs_names[i] == inputs_names[j]) {
            VLOG(7) << " ==== Custom Operator: " << op_type << "_grad "
                    << "'s No." << j << " inputs: " << inputs_names[j]
                    << " related to No." << i
                    << " grad_inputs fwd inputs: " << grad_inputs_names[i];
            in_out_map[op_type][1][3][j] = i;  // NOLINT
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
                      common::errors::NotFound(
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
        in_out_map[op_type][1][4][j] = i;  // NOLINT
      }
    }
  }
}

paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize>
RunCustomOpNode::operator()(paddle::small_vector<std::vector<paddle::Tensor>,
                                                 kSlotSmallVectorSize>& grads,
                            bool create_graph,
                            bool is_new_grad) {  // NOLINT
  paddle::CustomOpKernelContext ctx;
  const auto& meta_info_map = egr::Controller::Instance().GetOpMetaInfoMap();
  const auto& vec_map = meta_info_map.at(op_type_);
  const auto& grad_inputs_name =
      paddle::OpMetaInfoHelper::GetInputs(vec_map[1]);
  const auto& grad_outputs_names =
      paddle::OpMetaInfoHelper::GetOutputs(vec_map[1]);
  const auto& map =
      egr::Controller::Instance().GetCustomEdgesSlotMap().at(op_type_);

  paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize>
      tmp_ins(grad_inputs_name.size());
  VLOG(7) << " Prepare Backward inputs of grads with size: " << grads.size()
          << ", whose grad_inputs_name size is: " << grad_inputs_name.size();
  auto hooked_grads = ApplyGradientHooks(grads);
  for (size_t i = 0; i < hooked_grads.size(); i++) {
    if (map[0][1].find(static_cast<int>(i)) != map[0][1].end()) {
      VLOG(7) << "Insert grad: " << i
              << " to grad_inputs: " << map[0][1].at(static_cast<int>(i));
      tmp_ins[map[0][1].at(static_cast<int>(i))] = hooked_grads[i];
    }
  }

  for (auto it : fwd_outs) {  // NOLINT
    VLOG(7) << "Insert fwd_outs to grad_inputs: " << it.first;
    tmp_ins[it.first] = RunCustomOpNode::Recover(&(it.second));
  }

  for (auto it : fwd_ins) {  // NOLINT
    // NOTE(HongyuJia): returned tensor maybe un-defined tensor when inputs
    // optional<Tensor>
    VLOG(7) << "Insert fwd_ins to grad_inputs: " << it.first;
    tmp_ins[it.first] = RunCustomOpNode::Recover(&(it.second));
  }

  VLOG(6) << "Prepare Grad inputs";
  for (auto& in : tmp_ins) {
    for (auto& tensor : in) {
      if (tensor.initialized() && tensor.is_dense_tensor() &&
          !std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl())
               ->meta()
               .is_contiguous()) {
        tensor.set_impl(std::make_shared<phi::DenseTensor>(
            paddle::experimental::Trans2Contiguous(*(
                std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl())))));
      }
    }

    ctx.EmplaceBackInputs(in);
  }
  VLOG(6) << "Prepare Grad attrs";
  ctx.EmplaceBackAttrs(attrs_);
  // NOTE(HongyuJia): grad_outputs_names.size() <= OutputMeta().size():
  // OutputMeta().size() indicates input size of forward op,
  // grad_outputs_names.size() indicates output size of backward op.
  paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize> outs(
      OutputMeta().size());
  paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize>
      tmp_outs(grad_outputs_names.size());
  VLOG(6) << "Prepare Grad outputs for size: " << grad_outputs_names.size();
  for (size_t i = 0; i < OutputMeta().size(); i++) {
    if (map[0][0].find(static_cast<int>(i)) != map[0][0].end()) {
      int grad_output_idx = map[0][0].at(static_cast<int>(i));
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

  run_custom_op_impl(vec_map[1], false, false, ctx);

  for (size_t i = 0; i < ctx.OutputRange().size(); ++i) {
    auto output_pair = ctx.OutputRangeAt(i);
    outs[i] = ctx.OutputsBetween(output_pair.first, output_pair.second);
  }

  // handle optional None output when construct backward graph
  for (size_t i = 0; i < ctx.OutputRange().size(); i++) {
    if (ctx.OutputRangeAt(i).first + 1 == ctx.OutputRangeAt(i).second) {
      paddle::Tensor* out_tensor =
          ctx.MutableOutputAt(ctx.OutputRangeAt(i).first);
      if (!out_tensor->initialized()) {
        PADDLE_ENFORCE(
            paddle::framework::detail::IsOptionalVar(
                grad_outputs_names.at(i)) ||
                out_tensor->is_dist_tensor(),
            common::errors::InvalidArgument(
                "Custom grad operator[%s]'s %d-th output is not initialized. "
                "Please check your implementation again. If you are "
                "using inplace optional outputs, then you must use "
                "`paddle::Optional` to decorate this output",
                op_type_,
                i));
        // We can also consider using `autograd_meta` to tolerant nullptr.
        out_tensor->set_autograd_meta(std::make_shared<egr::AutogradMeta>());
      }
    }
  }

  VLOG(7) << "Get AutogradMeta for inputs and outputs for Custom grad Op";
  size_t slot_ins_num = ctx.InputRange().size();
  size_t slot_outs_num = ctx.OutputRange().size();
  VLOG(7) << "We got slot num of ins is: " << slot_ins_num;
  VLOG(7) << "We got slot num of outs is: " << slot_outs_num;
  std::vector<egr::AutogradMeta*> ins_auto_grad_metas =
      egr::EagerUtils::nullable_autograd_meta(*ctx.AllMutableInput());
  std::vector<egr::AutogradMeta*> outs_auto_grad_metas =
      egr::EagerUtils::unsafe_autograd_meta(*ctx.AllMutableOutput());

  bool require_any_grad = false;
  bool trace_backward = egr::Controller::Instance().HasGrad() && create_graph;
  for (auto& ins_auto_grad_meta : ins_auto_grad_metas) {
    require_any_grad =
        require_any_grad ||
        egr::EagerUtils::ComputeRequireGrad(trace_backward, ins_auto_grad_meta);
  }

  if (require_any_grad && (vec_map.size() > 2)) {
    phi::RecordEvent node_creation_record_event(
        "Custom Op " + op_type_ + " double_grad node_creation",
        phi::TracerEventType::OperatorInner,
        1);
    VLOG(6) << " Construct Grad for Custom Op: " << op_type_;
    ConstructFwdAndBwdMap(vec_map, op_type_);
    for (auto& outs_auto_grad_meta : outs_auto_grad_metas) {
      egr::EagerUtils::PassStopGradient(false, outs_auto_grad_meta);
    }
    // NOTE(HongyuJia): Does here needs to be consistent with forward process,
    // PassStopGradient to ins_auto_grad_metas?
    auto grad_node = std::make_shared<egr::RunCustomOpDoubleGradNode>(
        slot_outs_num, slot_ins_num, op_type_);

    const auto& slot_map = map;
    // Prepare Grad outputs
    size_t no_grad_cnt = 0;
    for (size_t i = 0; i < slot_ins_num; i++) {
      const std::vector<paddle::Tensor>& in_tensors = ctx.InputsBetween(
          ctx.InputRangeAt(i).first, ctx.InputRangeAt(i).second);

      if (slot_map[1][0].find(static_cast<int>(i)) != slot_map[1][0].end()) {
        grad_node->SetGradOutMeta(in_tensors,
                                  slot_map[1][0].at(static_cast<int>(i)));
      } else {
        grad_node->SetGradOutMeta(in_tensors, slot_ins_num - 1 - no_grad_cnt);
        no_grad_cnt++;
      }
    }

    // Prepare Grad inputs with grad of fwd outputs
    for (size_t i = 0; i < slot_outs_num; i++) {
      const auto& size_pair = ctx.OutputRangeAt(i);
      const std::vector<paddle::Tensor>& out_tensors =
          ctx.OutputsBetween(size_pair.first, size_pair.second);
      for (size_t j = size_pair.first; j < size_pair.second; j++) {
        // SetOutRankWithSlot: slot_id = i, rank = j - size_pair.first
        outs_auto_grad_metas[j]->SetSingleOutRankWithSlot(i,
                                                          j - size_pair.first);
        egr::EagerUtils::SetHistory(outs_auto_grad_metas[j], grad_node);
      }
      grad_node->SetGradInMeta(out_tensors, i);
    }

    // Prepare Grad inputs with fwd outputs
    for (auto item : slot_map[1][2]) {
      VLOG(7) << "Prepare fwd_outs: " << item.first
              << " to grad_inputs: " << item.second;
      grad_node->fwd_outs[item.second] =
          egr::RunCustomOpNode::ConstructTensorWrapper(
              ctx.OutputsBetween(ctx.OutputRangeAt(item.first).first,
                                 ctx.OutputRangeAt(item.first).second));
    }

    // Prepare Grad inputs with fwd inputs
    for (auto item : slot_map[1][3]) {
      VLOG(7) << "Prepare fwd_ins: " << item.first
              << " to grad_inputs: " << item.second;
      grad_node->fwd_ins[item.second] =
          egr::RunCustomOpNode::ConstructTensorWrapper(
              ctx.InputsBetween(ctx.InputRangeAt(item.first).first,
                                ctx.InputRangeAt(item.first).second));
    }

    std::vector<paddle::any> attrs(attrs_.size());
    // Prepare attrs for Grad node
    for (auto item : slot_map[1][4]) {
      VLOG(7) << "Prepare fwd attrs: " << item.first
              << " to grad_attrs: " << item.second;
      attrs[item.second] = attrs_[item.first];
    }
    grad_node->SetAttrs(attrs);
  }

  return outs;
}

paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize>
RunCustomOpDoubleGradNode::operator()(
    paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize>&
        grads,
    bool create_graph,
    bool is_new_grad) {  // NOLINT
  paddle::CustomOpKernelContext ctx;
  const auto& meta_info_map = egr::Controller::Instance().GetOpMetaInfoMap();
  const auto& vec_map = meta_info_map.at(op_type_);
  const auto& grad_inputs_name =
      paddle::OpMetaInfoHelper::GetInputs(vec_map[2]);
  const auto& grad_outputs_names =
      paddle::OpMetaInfoHelper::GetOutputs(vec_map[2]);
  const auto& map =
      egr::Controller::Instance().GetCustomEdgesSlotMap().at(op_type_);

  paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize>
      tmp_ins(grad_inputs_name.size());
  VLOG(7) << " Prepare Backward inputs of grads with size: " << grads.size()
          << ", whose grad_inputs_name size is: " << grad_inputs_name.size();

  auto hooked_grads = ApplyGradientHooks(grads);

  for (size_t i = 0; i < hooked_grads.size(); i++) {
    if (map[1][1].find(static_cast<int>(i)) != map[1][1].end()) {
      VLOG(7) << "Insert grad: " << i
              << " to grad_inputs: " << map[1][1].at(static_cast<int>(i));
      tmp_ins[map[1][1].at(static_cast<int>(i))] = hooked_grads[i];
    }
  }

  for (auto it : fwd_outs) {  // NOLINT
    VLOG(7) << "Insert fwd_outs to grad_inputs: " << it.first;
    tmp_ins[it.first] = RunCustomOpDoubleGradNode::Recover(&(it.second));
  }

  for (auto it : fwd_ins) {  // NOLINT
    VLOG(7) << "Insert fwd_ins to grad_inputs: " << it.first;
    tmp_ins[it.first] = RunCustomOpDoubleGradNode::Recover(&(it.second));
  }

  VLOG(6) << "Prepare Grad inputs";
  for (const auto& in : tmp_ins) {
    ctx.EmplaceBackInputs(in);
  }
  VLOG(6) << "Prepare Grad attrs";
  ctx.EmplaceBackAttrs(attrs_);
  paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize> outs(
      OutputMeta().size());
  paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize>
      tmp_outs(grad_outputs_names.size());
  VLOG(6) << "Prepare Grad outputs for size: " << grad_outputs_names.size();

  for (size_t i = 0; i < OutputMeta().size(); i++) {
    if (map[1][0].find(static_cast<int>(i)) != map[1][0].end()) {
      int grad_output_idx = map[1][0].at(static_cast<int>(i));
      VLOG(7) << "Insert grad outputs: " << i
              << " with size: " << OutputMeta()[grad_output_idx].size()
              << " to tmp_outputs: " << grad_output_idx;
      for (size_t j = 0; j < OutputMeta()[grad_output_idx].size(); j++) {
        outs[grad_output_idx]
            .emplace_back(/* init it in case of copy nullptr of shared_ptr */
                          std::make_shared<phi::DenseTensor>(
                              phi::DataType::UNDEFINED),
                          egr::Controller::Instance().GenerateUniqueName(
                              "custom_tmp_grad"));
      }
      tmp_outs[grad_output_idx] = outs[grad_output_idx];
    }
  }
  for (size_t i = 0; i < tmp_outs.size(); i++) {
    VLOG(7) << "Prepare grad outputs size: " << tmp_outs[i].size();
    ctx.EmplaceBackOutputs(tmp_outs[i]);
  }
  VLOG(7) << "Run Kernel of Grad Custom Op: " << op_type_ << "_grad_grad";

  run_custom_op_impl(vec_map[2], false, true, ctx);

  for (size_t i = 0; i < ctx.OutputRange().size(); ++i) {
    auto output_pair = ctx.OutputRangeAt(i);
    outs[i] = ctx.OutputsBetween(output_pair.first, output_pair.second);
  }

  return outs;
}
}  // namespace egr
