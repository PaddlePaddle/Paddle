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

#include "glog/logging.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/nan_inf_utils.h"
#include "paddle/fluid/eager/to_static/run_program_op_node.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/api/backward/backward_api.h"
#include "paddle/phi/api/backward/sparse_bw_api.h"

#include "paddle/fluid/eager/api/manual/eager_manual/nodes/nodes.h"
#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/phi/core/flags.h"

PHI_DECLARE_bool(check_nan_inf);

paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
Conv2dGradNodeFinal::operator()(
    paddle::small_vector<std::vector<paddle::Tensor>,
                         egr::kSlotSmallVectorSize>& grads,
    bool create_graph,
    bool is_new_grad) {
  // Fill Zero For GradIn Tensors
  VLOG(3) << " Running Conv2dGradNodeFinal: " << this;
  // Apply Gradient Hooks
  auto hooked_grads = ApplyGradientHooks(grads);

  // Collect GradIn Tensors, Attrs and Recovered TensorWrappers
  auto input = egr::EagerUtils::RecoverTensorWrapper(&this->input_);
  auto filter = egr::EagerUtils::RecoverTensorWrapper(&this->filter_);
  auto& grad_out = hooked_grads[0][0];
  auto& strides = this->strides_;
  auto& paddings = this->paddings_;
  auto& padding_algorithm = this->padding_algorithm_;
  auto& groups = this->groups_;
  auto& dilations = this->dilations_;
  auto& data_format = this->data_format_;
  // Prepare Grad function call

  const auto& out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
      returns(2);
  for (int i = 0; i < 2; ++i) {
    out_metas[i].empty() ? returns[i].resize(1)
                         : returns[i].resize(out_metas[i].size());
  }

  auto* api_output_0 =
      (out_metas[0].empty() || out_metas[0][0].IsStopGradient())
          ? nullptr
          : &returns[0][0];
  auto* api_output_1 =
      (out_metas[1].empty() || out_metas[1][0].IsStopGradient())
          ? nullptr
          : &returns[1][0];
  // Runtime check if we need next grad
  bool trace_backward = egr::Controller::Instance().HasGrad() && create_graph;

  // Inplace Check

  // Inplace Strategy


  // Call grad_api function
  VLOG(3) << "Final State Running: Conv2dGradNodeFinal";

  paddle::experimental::conv2d_grad(input,
                                    filter,
                                    grad_out,
                                    strides,
                                    paddings,
                                    padding_algorithm,
                                    dilations,
                                    groups,
                                    data_format,
                                    api_output_0,
                                    api_output_1);
  // Check NaN and Inf id needed
  if (FLAGS_check_nan_inf) {
    egr::CheckTensorHasNanOrInf("conv2d_grad", returns);
  }

  // Get GradOut autograd_meta

  auto& grad_input = returns[0][0];
  egr::AutogradMeta* grad_input_autograd_meta =
      returns[0][0].initialized() ? egr::EagerUtils::autograd_meta(&grad_input)
                                  : nullptr;
  if (grad_input_autograd_meta)
    grad_input_autograd_meta->SetStopGradient(false);
  VLOG(3) << "Conv2dGradNodeFinal grad_input_autograd_meta: "
          << grad_input_autograd_meta;

  auto& grad_filter = returns[1][0];
  egr::AutogradMeta* grad_filter_autograd_meta =
      returns[1][0].initialized() ? egr::EagerUtils::autograd_meta(&grad_filter)
                                  : nullptr;
  if (grad_filter_autograd_meta)
    grad_filter_autograd_meta->SetStopGradient(false);
  VLOG(3) << "Conv2dGradNodeFinal grad_filter_autograd_meta: "
          << grad_filter_autograd_meta;

  // Create Grad Node
  if (trace_backward) {
    paddle::platform::RecordEvent node_creation_record_event(
        "conv2d_grad node_creation",
        paddle::platform::TracerEventType::OperatorInner,
        1);

    // Node Construction
    auto grad_node = std::shared_ptr<Conv2dDoubleGradNodeFinal>(
        new Conv2dDoubleGradNodeFinal(2, 3));
    // SetAttributes if needed
    grad_node->SetAttributestrides(strides);
    grad_node->SetAttributepaddings(paddings);
    grad_node->SetAttributepadding_algorithm(padding_algorithm);
    grad_node->SetAttributegroups(groups);
    grad_node->SetAttributedilations(dilations);
    grad_node->SetAttributedata_format(data_format);
    // Set TensorWrappers for Forward Inputs if needed
    grad_node->SetTensorWrapperinput(input);
    grad_node->SetTensorWrapperfilter(filter);
    grad_node->SetTensorWrappergrad_out(grad_out);
    // SetGradOutMeta & SetEdges
    if (grad_filter_autograd_meta) {
      grad_node->SetGradOutMeta(input, 0);
    }
    if (grad_input_autograd_meta) {
      grad_node->SetGradOutMeta(filter, 1);
      grad_node->SetGradOutMeta(grad_out, 2);
    }
    // SetOutRank & SetHistory & SetGradInMeta & RetainGrad
    if (grad_input_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(grad_input_autograd_meta, 0);
    }
    if (grad_filter_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(grad_filter_autograd_meta, 1);
    }
    if (grad_input_autograd_meta) {
      egr::EagerUtils::SetHistory(grad_input_autograd_meta, grad_node);
    }
    if (grad_filter_autograd_meta) {
      egr::EagerUtils::SetHistory(grad_filter_autograd_meta, grad_node);
    }
    grad_node->SetGradInMeta(grad_input, 0);
    grad_node->SetGradInMeta(grad_filter, 1);
    // Set TensorWrappers for Forward Outputs if needed
  }

    VLOG(6) << "gradnode_ptr = " << this;
    if (VLOG_IS_ON(6)) {
    const char *INPUT_PRINT_TEMPLATE = "{ Input: [%s],  \n Output: [%s] } ";
    std::string input_str = "";
    std::string output_str = "";

    const char *TENSOR_INPUT_TEMPLATE = " \n( input , [%s]), ";
    std::string input_input_str =
        paddle::string::Sprintf(TENSOR_INPUT_TEMPLATE, egr::EagerUtils::TensorStr(input));
    input_str += input_input_str;

    const char *TENSOR_FILTER_TEMPLATE = " \n( filter , [%s]), ";
    std::string input_filter_str =
        paddle::string::Sprintf(TENSOR_FILTER_TEMPLATE, egr::EagerUtils::TensorStr(filter));
    input_str += input_filter_str;

    const char *TENSOR_GRAD_OUT_TEMPLATE = " \n( grad_out , [%s]), ";
    std::string input_grad_out_str =
        paddle::string::Sprintf(TENSOR_GRAD_OUT_TEMPLATE, egr::EagerUtils::TensorStr(grad_out));
    input_str += input_grad_out_str;

    const char *TENSOR_GRAD_INPUT_TEMPLATE = " \n ( grad_input , [%s]), ";
    std::string output_grad_input_str = paddle::string::Sprintf(
        TENSOR_GRAD_INPUT_TEMPLATE, egr::EagerUtils::TensorStr(grad_input));
    output_str += output_grad_input_str;

    const char *TENSOR_GRAD_FILTER_TEMPLATE = " \n ( grad_filter , [%s]), ";
    std::string output_grad_filter_str = paddle::string::Sprintf(
        TENSOR_GRAD_FILTER_TEMPLATE, egr::EagerUtils::TensorStr(grad_filter));
    output_str += output_grad_filter_str;

    VLOG(6) << paddle::string::Sprintf(INPUT_PRINT_TEMPLATE, input_str, output_str);
  }


  // Return
  if (NeedComplexToRealConversion()) HandleComplexGradToRealGrad(&returns);
  return returns;
}

paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
Conv2dDoubleGradNodeFinal::operator()(
    paddle::small_vector<std::vector<paddle::Tensor>,
                         egr::kSlotSmallVectorSize>& grads,
    bool create_graph,
    bool is_new_grad) {
  // Fill Zero For GradIn Tensors
  const auto& input_metas = this->InputMeta();
  egr::EagerUtils::FillZeroForEmptyOptionalGradInput(&grads[0][0],
                                                     input_metas[0][0]);
  egr::EagerUtils::FillZeroForEmptyOptionalGradInput(&grads[1][0],
                                                     input_metas[1][0]);

  // Apply Gradient Hooks
  auto hooked_grads = ApplyGradientHooks(grads);

  // Collect GradIn Tensors, Attrs and Recovered TensorWrappers
  auto input = egr::EagerUtils::RecoverTensorWrapper(&this->input_);
  auto filter = egr::EagerUtils::RecoverTensorWrapper(&this->filter_);
  auto grad_out = egr::EagerUtils::RecoverTensorWrapper(&this->grad_out_);
  auto& grad_input_grad = hooked_grads[0][0];

  paddle::optional<paddle::Tensor> grad_input_grad_optional;
  if (grad_input_grad.initialized())
    grad_input_grad_optional =
        paddle::make_optional<paddle::Tensor>(grad_input_grad);

  auto& grad_filter_grad = hooked_grads[1][0];

  paddle::optional<paddle::Tensor> grad_filter_grad_optional;
  if (grad_filter_grad.initialized())
    grad_filter_grad_optional =
        paddle::make_optional<paddle::Tensor>(grad_filter_grad);

  auto& strides = this->strides_;
  auto& paddings = this->paddings_;
  auto& padding_algorithm = this->padding_algorithm_;
  auto& groups = this->groups_;
  auto& dilations = this->dilations_;
  auto& data_format = this->data_format_;
  // Prepare Grad function call

  const auto& out_metas = OutputMeta();
  paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
      returns(3);
  for (int i = 0; i < 3; ++i) {
    out_metas[i].empty() ? returns[i].resize(1)
                         : returns[i].resize(out_metas[i].size());
  }

  auto* api_output_0 =
      (out_metas[0].empty() || out_metas[0][0].IsStopGradient())
          ? nullptr
          : &returns[0][0];
  auto* api_output_1 =
      (out_metas[1].empty() || out_metas[1][0].IsStopGradient())
          ? nullptr
          : &returns[1][0];
  auto* api_output_2 =
      (out_metas[2].empty() || out_metas[2][0].IsStopGradient())
          ? nullptr
          : &returns[2][0];
  // Runtime check if we need next grad

  // Inplace Check

  // Inplace Strategy

  // Call grad_api function
  VLOG(3) << "Final State Running: Conv2dGradGradNodeFinal";

  paddle::experimental::conv2d_grad_grad(input,
                                         filter,
                                         grad_out,
                                         grad_input_grad_optional,
                                         grad_filter_grad_optional,
                                         strides,
                                         paddings,
                                         padding_algorithm,
                                         dilations,
                                         groups,
                                         data_format,
                                         api_output_0,
                                         api_output_1,
                                         api_output_2);
  // Check NaN and Inf id needed
  if (FLAGS_check_nan_inf) {
    egr::CheckTensorHasNanOrInf("conv2d_grad_grad", returns);
  }

  // Get GradOut autograd_meta

  auto& input_grad = returns[0][0];
  egr::AutogradMeta* input_grad_autograd_meta =
      returns[0][0].initialized() ? egr::EagerUtils::autograd_meta(&input_grad)
                                  : nullptr;
  if (input_grad_autograd_meta)
    input_grad_autograd_meta->SetStopGradient(false);

  auto& filter_grad = returns[1][0];
  egr::AutogradMeta* filter_grad_autograd_meta =
      returns[1][0].initialized() ? egr::EagerUtils::autograd_meta(&filter_grad)
                                  : nullptr;
  if (filter_grad_autograd_meta)
    filter_grad_autograd_meta->SetStopGradient(false);

  auto& grad_out_grad = returns[2][0];
  egr::AutogradMeta* grad_out_grad_autograd_meta =
      returns[2][0].initialized()
          ? egr::EagerUtils::autograd_meta(&grad_out_grad)
          : nullptr;
  if (grad_out_grad_autograd_meta)
    grad_out_grad_autograd_meta->SetStopGradient(false);

  // Create Grad Node


   VLOG(6) << "gradnode_ptr = " << this;
  if (VLOG_IS_ON(6)) {
    const char *INPUT_PRINT_TEMPLATE = "{ Input: [%s],  \n Output: [%s] } ";
    std::string input_str = "";
    std::string output_str = "";

    const char *TENSOR_INPUT_TEMPLATE = " \n( input , [%s]), ";
    std::string input_input_str =
        paddle::string::Sprintf(TENSOR_INPUT_TEMPLATE, egr::EagerUtils::TensorStr(input));
    input_str += input_input_str;

    const char *TENSOR_FILTER_TEMPLATE = " \n( filter , [%s]), ";
    std::string input_filter_str =
        paddle::string::Sprintf(TENSOR_FILTER_TEMPLATE, egr::EagerUtils::TensorStr(filter));
    input_str += input_filter_str;

    const char *TENSOR_GRAD_OUT_TEMPLATE = " \n( grad_out , [%s]), ";
    std::string input_grad_out_str =
        paddle::string::Sprintf(TENSOR_GRAD_OUT_TEMPLATE, egr::EagerUtils::TensorStr(grad_out));
    input_str += input_grad_out_str;

    const char *TENSOR_GRAD_INPUT_GRAD_TEMPLATE = " \n( grad_input_grad , [%s]), ";
    std::string input_grad_input_grad_str =
        paddle::string::Sprintf(TENSOR_GRAD_INPUT_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(grad_input_grad));
    input_str += input_grad_input_grad_str;

    const char *TENSOR_GRAD_FILTER_GRAD_TEMPLATE = " \n( grad_filter_grad , [%s]), ";
    std::string input_grad_filter_grad_str =
        paddle::string::Sprintf(TENSOR_GRAD_FILTER_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(grad_filter_grad));
    input_str += input_grad_filter_grad_str;

    const char *TENSOR_INPUT_GRAD_TEMPLATE = " \n( input_grad , [%s]), ";
    std::string output_input_grad_str =
        paddle::string::Sprintf(TENSOR_INPUT_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(input_grad));
    output_str += output_input_grad_str;

    const char *TENSOR_FILTER_GRAD_TEMPLATE = " \n( filter_grad , [%s]), ";
    std::string output_filter_grad_str =
        paddle::string::Sprintf(TENSOR_FILTER_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(filter_grad));
    output_str += output_filter_grad_str;

    const char *TENSOR_GRAD_OUT_GRAD_TEMPLATE = " \n( grad_out_grad , [%s]) ";
    std::string output_grad_out_grad_str =
        paddle::string::Sprintf(TENSOR_GRAD_OUT_GRAD_TEMPLATE, egr::EagerUtils::TensorStr(grad_out_grad));
    output_str += output_grad_out_grad_str;

    VLOG(6) << paddle::string::Sprintf(INPUT_PRINT_TEMPLATE, input_str, output_str);
  }


  // Return
  if (NeedComplexToRealConversion()) HandleComplexGradToRealGrad(&returns);
  return returns;
}
