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

#include "paddle/common/flags.h"
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
#include "paddle/fluid/eager/api/manual/eager_manual/nodes/nodes.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/eager_layout_auto_tune.h"
#include "paddle/fluid/eager/nan_inf_utils.h"
#include "paddle/fluid/imperative/amp_utils.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"
#define SEPARATOR "=========================="

COMMON_DECLARE_bool(check_nan_inf);
COMMON_DECLARE_bool(check_cuda_error);
COMMON_DECLARE_bool(enable_unique_name);

paddle::Tensor conv2d_ad_func(
    const paddle::Tensor& input,
    const paddle::Tensor& filter,
    std::vector<int> strides,
    std::vector<int> paddings,
    std::string padding_algorithm,
    std::vector<int> dilations,
    int groups,
    std::string data_format,
    paddle::optional<paddle::Tensor*> predefined_out) {
  VLOG(3) << "\n"
          << SEPARATOR << "Running_AD_API: "
          << "conv2d" << SEPARATOR;
  if (FLAGS_check_cuda_error) [[unlikely]] {
    egr::CUDAErrorCheck("conv2d_ad_func begin");
  }
  // Dygraph Record Event
  phi::RecordEvent dygraph_entrance_record_event(
      "conv2d dygraph", phi::TracerEventType::Operator, 1);

  // AMP Logic
  if (egr::Controller::Instance().GetAMPLevel() !=
      paddle::imperative::AmpLevel::O0) {
    VLOG(5) << "Check and Prepare For AMP";
    auto op_name = phi::TransToFluidOpName("conv2d");
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        amp_tensors_vector = {{input}, {filter}};

    auto amp_dst_dtype =
        paddle::imperative::GetAmpDestDtype(op_name, amp_tensors_vector);
    VLOG(5) << "AMP Get Dest Dtype : " << amp_dst_dtype;
    auto new_input =
        paddle::imperative::AmpAutoCast("input", input, amp_dst_dtype, op_name);
    auto new_filter = paddle::imperative::AmpAutoCast(
        "filter", filter, amp_dst_dtype, op_name);

    {
      paddle::imperative::AutoCastGuard guard(
          egr::Controller::Instance().GetCurrentAmpAttrs(),
          paddle::imperative::AmpLevel::O0);
      return conv2d_ad_func(new_input,
                            new_filter,
                            strides,
                            paddings,
                            padding_algorithm,
                            dilations,
                            groups,
                            data_format);
    }
  }

  // Layout autotune

  if (egr::Controller::Instance().UseLayoutAutoTune()) {
    VLOG(5) << "Check and Prepare For LAYOUT";
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        tensors_vector = {{input}, {filter}};

    auto op_name = phi::TransToFluidOpName("conv2d");
    auto transformer = egr::EagerLayoutAutotune<std::string>(
        op_name, tensors_vector, &data_format);
    auto new_input = transformer->TransInTensor("input", input);
    bool need_tune = egr::Controller::Instance().UseLayoutAutoTune();
    egr::Controller::Instance().DisableLayoutAutoTune();
    auto out = conv2d_ad_func(new_input,
                              filter,
                              strides,
                              paddings,
                              padding_algorithm,
                              dilations,
                              groups,
                              data_format);
    transformer->SetOutTensorLayout(&out);
    if (need_tune) {
      egr::Controller::Instance().EnableLayoutAutoTune();
    }
    // Returns
    return out;
  }

  // Get Input AutoGradMeta
  egr::AutogradMeta* input_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(input);
  egr::AutogradMeta* filter_autograd_meta =
      egr::EagerUtils::nullable_autograd_meta(filter);
  // Forward API Call
  std::string unique_api_name;
  if (VLOG_IS_ON(3) || FLAGS_enable_unique_name) {
    static int64_t call_count = 0;
    call_count++;
    unique_api_name = egr::GenerateUniqueApiName("conv2d", call_count);
  }
  VLOG(3) << "\n"
          << SEPARATOR << "Running_C++_API: " << unique_api_name << SEPARATOR;
  auto api_result = paddle::experimental::conv2d(input,
                                                 filter,
                                                 strides,
                                                 paddings,
                                                 padding_algorithm,
                                                 dilations,
                                                 groups,
                                                 data_format);
  VLOG(3) << "\n"
          << SEPARATOR << "Finshi_C++_API: " << unique_api_name << SEPARATOR;
  // Check NaN and Inf if needed
  if (FLAGS_check_nan_inf) {
    egr::CheckTensorHasNanOrInf("conv2d", api_result);
  }

  // Get Outputs
  auto& out = api_result;
  if (VLOG_IS_ON(6) || FLAGS_enable_unique_name) {
    egr::SetTensorName(unique_api_name, "out", &out);
  }

  // Get Output AutoGradMeta
  egr::AutogradMeta* out_autograd_meta = egr::EagerUtils::autograd_meta(&out);
  bool trace_backward = egr::Controller::Instance().HasGrad();
  bool require_any_grad = egr::EagerUtils::ComputeRequireGrad(
      trace_backward, input_autograd_meta, filter_autograd_meta);

  // Check Inplace if needed

  // Node Creation
  if (require_any_grad) {
    phi::RecordEvent node_creation_record_event(
        "conv2d node_creation", phi::TracerEventType::OperatorInner, 1);

    egr::EagerUtils::PassStopGradient(false, out_autograd_meta);

    // Node Construction
    auto grad_node = std::shared_ptr<Conv2dGradNodeFinal>(  // NOLINT
        new Conv2dGradNodeFinal(1, 2));
    // Set GradNodeName
    if (VLOG_IS_ON(6) || FLAGS_enable_unique_name) {
      grad_node->SetNameFromAPI(unique_api_name);
    }
    // Set forward's stack
    if (FLAGS_check_nan_inf) {
      grad_node->SetForwardTrace(egr::Controller::Instance().GetPythonStack());
    }
    // Set for Record Subgraph
    if (egr::EagerBackwardSubGraphNodeRecorder::Instance()
            .NeedCaptureSubGraph()) {
      VLOG(3) << "Capture the grad node" << grad_node->name() << "("
              << grad_node.get() << ")"
              << "for subgraph.";
      egr::EagerBackwardSubGraphNodeRecorder::Instance().AddGradNode(
          grad_node.get());
    }
    // SetAttributes if needed
    grad_node->SetAttribute_strides(strides);
    grad_node->SetAttribute_paddings(paddings);
    grad_node->SetAttribute_padding_algorithm(padding_algorithm);
    grad_node->SetAttribute_groups(groups);
    grad_node->SetAttribute_dilations(dilations);
    grad_node->SetAttribute_data_format(data_format);
    // Set TensorWrappers for Forward Inputs if needed
    grad_node->SetTensorWrapper_input(input);
    grad_node->SetTensorWrapper_filter(filter);
    // SetGradOutMeta & SetEdges
    grad_node->SetGradOutMeta(input, 0);
    grad_node->SetGradOutMeta(filter, 1);
    // SetOutRank & SetHistory & SetGradInMeta & RetainGrad
    if (out_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(out_autograd_meta, 0);
    }
    if (out_autograd_meta) {
      egr::EagerUtils::SetHistory(out_autograd_meta, grad_node);
    }
    grad_node->SetGradInMeta(out, 0);
    // Set TensorWrappers for Forward Outputs if needed
  }

  if (FLAGS_check_cuda_error) [[unlikely]] {
    egr::CUDAErrorCheck("conv2d_ad_func finish");
  }
  VLOG(3) << "\n"
          << SEPARATOR << "Finish_AD_API: "
          << "conv2d" << SEPARATOR;
  // Returns
  return out;
}
